# Warmup 阶段负 Reward 问题分析

## 问题

在整个 Warmup 阶段，观察到：
```
reward/discriminator_mean: -1.88
critic/rewards/mean: -1.88
```

两个指标都是负数，这是否正常？

## 答案

**✅ 完全正常！这是 Warmup 阶段的预期行为。**

## 两个指标的区别

### 1. `reward/discriminator_mean` - Critic 的原始打分

**定义**: Critic 对 **Student responses** 的原始评分（sequence-level score）

**计算位置**: `verl/verl/trainer/ppo/ray_trainer.py:1450`

```python
# Critic 对 student responses 打分
discriminator_reward = critic.compute_values(student_data)  # shape: (batch, seq_len)

# 计算每个样本的总分
disc_reward_sum = discriminator_reward.sum(dim=-1)  # shape: (batch,)

# 记录均值
format_metrics["reward/discriminator_mean"] = disc_reward_sum.mean().item()
```

**含义**:
- Critic 认为 student responses 的质量如何
- 在 Warmup 阶段，student 质量远低于 teacher，所以是负数

### 2. `critic/rewards/mean` - Actor 使用的最终 Reward

**定义**: Actor 真正用于策略优化的 reward（可能包含 format reward）

**计算位置**: `verl/verl/trainer/ppo/metric_utils.py:163`

```python
# 从 batch 中获取 token_level_rewards（已经组合了所有 reward）
sequence_reward = batch.batch["token_level_rewards"].sum(-1)  # shape: (batch,)

# 只统计非 aborted 样本
non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

# 记录均值
reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
metrics["critic/rewards/mean"] = reward_mean
```

**含义**:
- Actor 实际收到的奖励信号
- 用于计算 advantages 和 policy gradient
- 在 Warmup 阶段，如果没有 format reward，等于 `reward/discriminator_mean`

## 为什么在 Warmup 阶段都是负数？

### 原因 1: Critic 的判别逻辑

在 Warmup 阶段，Critic 学习区分 teacher 和 student：

```python
# Critic 的训练目标
teacher_score > student_score

# 实际训练结果 (Step 253)
teacher_score_mean: 1.81  (正数)
student_score_mean: -1.33 (负数)
```

**Critic 学到的模式**:
- Teacher responses → 正分 (好的回答)
- Student responses → 负分 (差的回答)

### 原因 2: 未训练的 Student

Warmup 阶段 **Actor 被冻结**，student 是未训练的基础模型：

```python
# Warmup 阶段配置
freeze_actor_steps = 253  # Actor 完全冻结

# 结果
student 质量 << teacher 质量
→ Critic 给 student 负分
→ reward/discriminator_mean < 0
→ critic/rewards/mean < 0
```

### 原因 3: 相对评分机制

Critic 使用 **相对评分**，不是绝对评分：

```python
# Discriminator loss
diff = teacher_score - student_score
ranking_loss = -log(sigmoid(diff / temperature))

# 训练目标
teacher_score > student_score  # 只要求相对大小

# 结果
teacher_score = 1.81   (正数，但不大)
student_score = -1.33  (负数，但不小)
diff = 3.14            (关键是差值)
```

**关键点**: Critic 不关心绝对分数，只关心 teacher 比 student 高多少。

## 这是否有问题？

### ✅ 完全正常，无需担心

**理由**:

1. **Warmup 阶段的目标**: 训练 Critic 学会区分 teacher 和 student
   - ✅ d_acc = 94.73% (Critic 能区分)
   - ✅ score_diff = 1.81 (teacher > student)
   - ✅ 目标达成

2. **负数不影响训练**: Critic 使用相对评分
   - 重要的是 `teacher_score - student_score > 0`
   - 不重要的是绝对值的正负

3. **Stage 2 会改善**: 进入 GAD 训练后
   - Actor 开始学习
   - Student 质量提升
   - `reward/discriminator_mean` 会逐渐上升
   - 最终可能变为正数

## 预期的训练轨迹

### Warmup 阶段 (当前)
```
Step 1-253:
  reward/discriminator_mean: -2.0 ~ -1.5 (负数)
  critic/rewards/mean: -2.0 ~ -1.5 (负数)
  
  原因: Student 未训练，质量差
  状态: ✅ 正常
```

### Stage 2 (GAD) 初期
```
Step 254-500:
  reward/discriminator_mean: -1.5 → -0.5 (逐渐上升)
  critic/rewards/mean: -1.5 → -0.5 (逐渐上升)
  
  原因: Actor 开始学习，Student 质量提升
  状态: ✅ 健康训练
```

### Stage 2 (GAD) 中期
```
Step 500-1000:
  reward/discriminator_mean: -0.5 → 0.5 (可能变正)
  critic/rewards/mean: -0.5 → 0.5 (可能变正)
  
  原因: Student 质量接近 Teacher
  状态: ✅ 训练成功
```

### Stage 2 (GAD) 后期
```
Step 1000+:
  reward/discriminator_mean: 0.5 ~ 1.0 (正数)
  critic/rewards/mean: 0.5 ~ 1.0 (正数)
  
  原因: Student 质量达到或超过 Teacher
  状态: ✅ 训练收敛
```

## 如何判断训练是否健康？

### ✅ 健康的 Warmup 训练

```python
# 关键指标
d_acc: 70-95%  # Critic 能区分 teacher 和student
score_diff: 1-5  # Teacher 明显优于 student
reward/discriminator_mean: 负数  # Student 质量差（正常）
critic/rewards/mean: 负数  # 与 discriminator_mean 一致

# 稳定性指标
grad_norm: 0.3-5.0  # 梯度健康
ranking_loss: 稳定  # 训练稳定
```

### ❌ 有问题的 Warmup 训练

```python
# 异常情况 1: Critic 无法区分
d_acc: < 60%  # Critic 学习失败
score_diff: < 0.5  # Teacher 和 student 无差异

# 异常情况 2: 分数异常
reward/discriminator_mean: > 5.0  # 分数过大
reward/discriminator_mean: < -10.0  # 分数过小

# 异常情况 3: 训练不稳定
grad_norm: > 10.0  # 梯度爆炸
ranking_loss: 剧烈震荡  # 训练不稳定
```

## 当前训练状态评估 (Step 253)

```python
✅ reward/discriminator_mean: -1.88
   - 负数是正常的（Student 质量差）
   - 绝对值不大（-1.88），说明 Critic 评分合理

✅ critic/rewards/mean: -1.88
   - 与 discriminator_mean 一致（没有 format reward）
   - Actor 会收到这个负奖励信号

✅ d_acc: 94.73%
   - Critic 能很好地区分 teacher 和 student

✅ score_diff: 1.81
   - Teacher 明显优于 student

✅ 训练稳定
   - grad_norm: 0.398 (健康)
   - ranking_loss: 0.530 (稳定)
```

**结论**: ✅ **训练完全正常，可以进入 Stage 2**

## 进入 Stage 2 后的预期

### 监控重点

```python
# 核心指标（应该逐渐上升）
critic/rewards/mean  # 从 -1.88 逐渐上升
reward/discriminator_mean  # 从 -1.88 逐渐上升

# 配合指标
actor/pg_loss  # 应该下降
critic/score_diff  # 应该缩小（student 接近 teacher）
format/reward_avg  # 如果有 format reward，应该提升
```

### 健康的 Stage 2 训练

```python
Step 254-300:
  critic/rewards/mean: -1.88 → -1.5 (上升)
  actor/pg_loss: 下降
  critic/score_diff: 1.81 → 1.5 (缩小)
  
Step 300-500:
  critic/rewards/mean: -1.5 → -0.5 (持续上升)
  actor/pg_loss: 持续下降
  critic/score_diff: 1.5 → 1.0 (持续缩小)
  
Step 500+:
  critic/rewards/mean: -0.5 → 0+ (可能变正)
  actor/pg_loss: 稳定在低位
  critic/score_diff: 1.0 → 0.5 (student 接近 teacher)
```

## 总结

### 核心要点

1. **负 reward 在 Warmup 阶段是正常的**
   - Student 未训练，质量差
   - Critic 正确地给了负分

2. **两个指标的关系**
   ```python
   # 如果没有 format reward
   critic/rewards/mean ≈ reward/discriminator_mean
   
   # 如果有 format reward
   critic/rewards/mean = reward/discriminator_mean + format_contribution
   ```

3. **判断训练健康的关键**
   - 不是看 reward 的正负
   - 而是看 d_acc, score_diff, 训练稳定性

4. **进入 Stage 2 后会改善**
   - Actor 开始学习
   - Student 质量提升
   - Reward 逐渐上升，可能变正

### 当前状态

**✅ Warmup 训练成功完成，可以进入 Stage 2 (GAD) 训练**

---

**创建日期**: 2026-01-28
**状态**: ✅ 问题已解答
