# Reward 指标说明

## 问题

`reward/discriminator_mean` 和 `critic/rewards/mean` 有什么区别？哪一个能够体现 Actor 的 reward？

## 答案

**`critic/rewards/mean` 才是 Actor 真正使用的 reward**，它是最终传递给 Actor 进行策略优化的奖励信号。

## 详细解释

### 1. `reward/discriminator_mean` - Discriminator 原始奖励

**定义位置**: `verl/verl/trainer/ppo/ray_trainer.py:1450`

**计算方式**:
```python
# GAD 模式：使用 critic values 作为 discriminator reward
discriminator_reward = batch.batch["values"]  # 来自 Critic 模型的输出

# 计算每个样本的总 reward（序列级别）
disc_reward_sum = discriminator_reward.sum(dim=-1)

# 记录均值
format_metrics["reward/discriminator_mean"] = disc_reward_sum.mean().item()
```

**含义**:
- 这是 **Critic 模型直接输出的原始分数**
- 在 GAD 模式下，Critic 对 Student 响应的质量评分
- **这是组合前的 discriminator reward**，还没有加上 format reward

**特点**:
- ✅ 反映 Critic 对响应质量的判断
- ✅ 用于监控 Critic 的打分行为
- ❌ **不是** Actor 最终使用的 reward（可能还会加上 format reward）

---

### 2. `critic/rewards/mean` - Actor 最终使用的 Reward

**定义位置**: `verl/verl/trainer/ppo/metric_utils.py:163`

**计算方式**:
```python
# 从 batch 中获取 token_level_rewards（这是最终传递给 Actor 的 reward）
sequence_reward = batch.batch["token_level_rewards"].sum(-1)

# 只统计非 aborted 样本
non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

# 记录均值
reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
metrics["critic/rewards/mean"] = reward_mean
```

**含义**:
- 这是 **Actor 在 PPO 更新时实际使用的 reward**
- 在 GAD 模式下，这是 `discriminator_reward + format_reward` 的组合
- 这是计算 advantages 和 returns 的基础

**特点**:
- ✅ **这是 Actor 真正优化的目标**
- ✅ 包含了所有 reward 组件（discriminator + format）
- ✅ 用于计算 advantages: `A = rewards + γV(s') - V(s)`
- ✅ 用于计算 returns: `R = rewards + γV(s')`

---

## 在 GAD 训练中的 Reward 流程

### 完整流程

```
1. Critic 打分
   ↓
   discriminator_reward = critic.values
   ↓
   记录: reward/discriminator_mean
   
2. 格式奖励计算（可选）
   ↓
   format_reward = compute_format_reward(responses)
   
3. Reward 组合
   ↓
   final_reward = discriminator_reward + format_weight * format_reward
   ↓
   batch.batch["token_level_rewards"] = final_reward
   
4. 传递给 Actor
   ↓
   Actor 使用 token_level_rewards 计算 advantages
   ↓
   记录: critic/rewards/mean
   
5. Actor 更新
   ↓
   policy_loss = -advantages * log_prob_ratio
```

### 代码对应关系

```python
# Step 1: Critic 打分（ray_trainer.py:1442）
discriminator_reward = batch.batch["values"]
format_metrics["reward/discriminator_mean"] = disc_reward_sum.mean().item()

# Step 2-3: Reward 组合（ray_trainer.py:1453-1477）
if format_reward_tensor is not None:
    format_contribution = format_weight * format_reward_tensor
    reward_tensor = discriminator_reward + format_contribution
    
    # 记录组合后的 reward
    format_metrics["reward/combined_mean"] = combined_reward_sum.mean().item()

# Step 4: 存储到 batch（ray_trainer.py 后续代码）
batch.batch["token_level_rewards"] = reward_tensor

# Step 5: Actor 使用（metric_utils.py:106）
sequence_reward = batch.batch["token_level_rewards"].sum(-1)
metrics["critic/rewards/mean"] = reward_mean
```

---

## 关键区别总结

| 指标 | 来源 | 含义 | 是否包含 format reward | Actor 是否使用 |
|------|------|------|----------------------|---------------|
| `reward/discriminator_mean` | Critic 输出 | Discriminator 原始评分 | ❌ 否 | ❌ 否 |
| `reward/format_contribution_mean` | Format reward 计算 | 格式奖励贡献 | N/A | ❌ 否 |
| `reward/combined_mean` | 组合后 | Discriminator + Format | ✅ 是 | ✅ 是（如果有 format reward）|
| `critic/rewards/mean` | `token_level_rewards` | **Actor 最终使用的 reward** | ✅ 是 | ✅ **是** |

---

## 监控建议

### 1. 监控 Actor 的学习效果

**关注指标**: `critic/rewards/mean`

```python
# 这个指标应该逐渐提升
critic/rewards/mean  # Actor 获得的平均 reward

# 配合其他指标判断
critic/advantages/mean  # 应该接近 0（GAE 归一化后）
critic/returns/mean    # 应该逐渐提升
actor/pg_loss          # 应该逐渐下降
```

**判断标准**:
- ✅ `critic/rewards/mean` 逐渐提升 → Actor 在进步
- ⚠️ `critic/rewards/mean` 停滞 → Actor 学习停滞
- ❌ `critic/rewards/mean` 下降 → Actor 质量退化

### 2. 监控 Reward 组成

**关注指标**: `reward/discriminator_mean`, `reward/format_contribution_mean`, `reward/combined_mean`

```python
# 查看 reward 的组成
reward/discriminator_mean        # Critic 评分部分
reward/format_contribution_mean  # 格式奖励部分
reward/combined_mean             # 组合后的总 reward
reward/format_ratio              # 格式奖励占比

# 验证一致性
# 如果有 format reward:
#   reward/combined_mean ≈ reward/discriminator_mean + reward/format_contribution_mean
#   critic/rewards/mean ≈ reward/combined_mean
# 如果没有 format reward:
#   critic/rewards/mean ≈ reward/discriminator_mean
```

**判断标准**:
- ✅ `reward/format_ratio` 在 0.1-0.3 → 格式奖励权重合理
- ⚠️ `reward/format_ratio` > 0.5 → 格式奖励权重过大，可能主导训练
- ❌ `reward/format_ratio` > 0.8 → 格式奖励完全主导，Critic 信号被淹没

### 3. 监控 Critic 的判别能力

**关注指标**: `reward/discriminator_mean` 配合 `critic/score_diff`

```python
# Critic 的判别能力
critic/d_acc                # 判别准确率（应该 65-85%）
critic/score_diff           # Teacher-Student 分数差异（应该逐渐缩小）
reward/discriminator_mean   # Discriminator 给 Student 的平均分

# 判断 Critic 是否有效
# 如果 Student 质量提升，discriminator_mean 应该提升
```

---

## 常见问题

### Q1: 为什么有两个 reward 指标？

**A**: 
- `reward/discriminator_mean` 用于监控 **Critic 的打分行为**
- `critic/rewards/mean` 用于监控 **Actor 的学习效果**

它们的关系：
```
critic/rewards/mean = reward/discriminator_mean + reward/format_contribution_mean
```

### Q2: 如果没有 format reward，两个指标是否相同？

**A**: 是的，如果没有 format reward：
```python
critic/rewards/mean ≈ reward/discriminator_mean
```

但它们的计算时机和统计范围可能略有不同：
- `reward/discriminator_mean`: 在 reward 计算阶段记录
- `critic/rewards/mean`: 在 data metrics 计算阶段记录，只统计非 aborted 样本

### Q3: Actor 优化时使用哪个 reward？

**A**: **Actor 使用 `critic/rewards/mean` 对应的 `token_level_rewards`**

具体流程：
```python
# 1. Reward 存储到 batch
batch.batch["token_level_rewards"] = final_reward

# 2. 计算 advantages（PPO 核心）
advantages = compute_gae(
    rewards=batch.batch["token_level_rewards"],  # ← 使用这个
    values=batch.batch["values"],
    ...
)

# 3. Actor 更新
policy_loss = -advantages * log_prob_ratio  # ← advantages 来自 token_level_rewards
```

### Q4: 如何判断 Actor 是否在有效学习？

**A**: 综合判断以下指标：

```python
# 核心指标（必看）
critic/rewards/mean  ↑  # 应该逐渐提升
actor/pg_loss        ↓  # 应该逐渐下降
critic/score_diff    ↓  # Student 与 Teacher 差距缩小

# 辅助指标
reward/discriminator_mean  ↑  # Critic 给 Student 的分数提升
format/reward_avg          ↑  # 格式质量提升
actor/approx_kl            ~  # 保持在合理范围（< 0.02）
```

**健康的训练**:
- ✅ `critic/rewards/mean` 稳定提升
- ✅ `actor/pg_loss` 稳定下降
- ✅ `critic/score_diff` 逐渐缩小
- ✅ `actor/approx_kl` 保持稳定

**有问题的训练**:
- ❌ `critic/rewards/mean` 停滞或下降
- ❌ `actor/pg_loss` 不下降或震荡
- ❌ `critic/score_diff` 不缩小
- ❌ `actor/approx_kl` 过大（> 0.05）

---

## TensorBoard 监控建议

### 关键曲线组合

```python
# 组 1: Actor 学习效果（最重要）
critic/rewards/mean      # 应该 ↑
actor/pg_loss            # 应该 ↓
critic/score_diff        # 应该 ↓

# 组 2: Reward 组成分析
reward/discriminator_mean
reward/format_contribution_mean
reward/combined_mean
reward/format_ratio

# 组 3: Critic 判别能力
critic/d_acc
critic/score_separation
reward/discriminator_mean

# 组 4: 训练稳定性
actor/approx_kl
actor/clipfrac
critic/vf_explained_var
```

---

## 总结

### 核心要点

1. **`critic/rewards/mean` 是 Actor 真正使用的 reward**
   - 包含 discriminator reward + format reward
   - 用于计算 advantages 和 returns
   - 反映 Actor 的学习效果

2. **`reward/discriminator_mean` 是 Critic 的原始评分**
   - 只包含 discriminator reward
   - 不包含 format reward
   - 用于监控 Critic 的打分行为

3. **监控 Actor 学习效果，看 `critic/rewards/mean`**
   - 应该逐渐提升
   - 配合 `actor/pg_loss` 和 `critic/score_diff` 判断

4. **监控 Reward 组成，看 `reward/*` 系列指标**
   - 了解 discriminator 和 format reward 的贡献
   - 确保 format reward 不会主导训练

---

**版本**: v1.0  
**更新日期**: 2026-01-27  
**作者**: AI Assistant
