# d_acc 持续高位问题深度诊断

## 问题现象

即使在引入随机化 forward 顺序后，`critic/d_acc` 仍然维持在 **96-99%** 的极高水平（从你的训练曲线可以看到）。

这说明问题**不仅仅是顺序依赖**，还有更深层的原因。

## 可能的根本原因分析

### 原因1：Teacher 和 Student 质量差距过大 ⭐⭐⭐⭐⭐

**最可能的原因**

#### 问题描述

如果 teacher response 的质量**远远超过** student response，那么即使 critic 正常学习，d_acc 也会自然地维持在很高水平。

#### 验证方法

```python
# 在训练日志中检查以下指标：
critic/score_diff          # 如果持续 > 1.0，说明差距很大
critic/teacher_value_mean  # Teacher 的平均分数
critic/student_value_mean  # Student 的平均分数
actor/format_reward_mean   # Student 的格式奖励

# 如果观察到：
# - score_diff 持续很大（例如 > 2.0）
# - teacher_value_mean >> student_value_mean
# - format_reward_mean 很低（例如 < -0.5）
# 说明质量差距确实很大
```

#### 为什么会这样？

1. **训练初期**：Student 模型刚开始训练，质量很差
2. **Teacher 质量高**：Teacher response 来自高质量数据或强模型
3. **Format reward 严格**：你的 GAD format reward 有很多惩罚项，student 容易触发

#### 解决方案

**方案A：调整 discriminator loss 的 temperature**

```python
# 在 core_algos.py 的 compute_discriminator_loss 中
# 当前值
temperature = 2.0

# 如果 d_acc 太高，增大 temperature
temperature = 5.0  # 或更大，例如 10.0

# Temperature 的作用：
# - 越大：loss 对差异越不敏感，d_acc 会下降
# - 越小：loss 对差异越敏感，d_acc 会上升
```

**方案B：使用 label smoothing**

```python
# 修改 discriminator loss 计算
def compute_discriminator_loss_with_smoothing(
    student_vpreds, teacher_vpreds, 
    response_mask, teacher_response_mask,
    label_smoothing=0.1  # 新增参数
):
    # ... 前面的计算相同 ...
    
    # 原始 ranking loss
    # ranking_loss = -log(sigmoid(diff))
    
    # 加入 label smoothing
    # 目标不是 100% 确定 teacher > student
    # 而是 (1-ε) 的概率 teacher > student，ε 的概率 student > teacher
    pos_loss = -torch.nn.functional.logsigmoid(scaled_diff)
    neg_loss = -torch.nn.functional.logsigmoid(-scaled_diff)
    ranking_loss = ((1 - label_smoothing) * pos_loss + label_smoothing * neg_loss).mean()
    
    # ... 后面的计算相同 ...
```

**方案C：降低 format reward 的惩罚力度**

```python
# 在 gad_format_reward.py 中
# 当前的惩罚可能过于严格，导致 student 分数过低

# 例如，减小惩罚系数：
def compute_format_score(solution_str, ground_truth=""):
    # ... 前面的检测逻辑相同 ...
    
    # 减小惩罚力度
    if json_issue:
        score -= json_issue["penalty"] * 0.5  # 乘以 0.5 减半惩罚
    
    if lang_issue:
        score -= lang_issue["penalty"] * 0.5
    
    if content_issue:
        score -= content_issue["penalty"] * 0.5
    
    # ... 后面的逻辑相同 ...
```

### 原因2：Critic 模型容量不足 ⭐⭐⭐

#### 问题描述

如果 critic 模型太小或训练不充分，它可能只学会了**粗粒度的区分**（teacher 明显好 vs student 明显差），而无法学习**细粒度的质量评估**。

#### 验证方法

```python
# 检查 critic 模型的参数量和训练步数
# 如果 critic 是一个很小的模型（例如 < 100M 参数）
# 或者训练步数很少（例如 < 1000 steps）
# 可能存在容量不足问题
```

#### 解决方案

1. **增加 critic 训练步数**：
```python
# 在配置中增加 ppo_epochs
ppo_epochs = 4  # 从 2 增加到 4
```

2. **使用更大的 critic 模型**：
```python
# 如果可能，使用与 actor 相同大小的模型作为 critic
```

### 原因3：数据分布问题 ⭐⭐⭐⭐

#### 问题描述

如果训练数据中 teacher 和 student 的质量差异**本身就很一致**（例如，所有样本的 teacher 都比 student 好很多），那么 d_acc 自然会很高。

#### 验证方法

```python
# 检查数据集
# 1. 随机抽取 10-20 个样本
# 2. 人工评估 teacher 和 student 的质量差异
# 3. 看是否所有样本都是 teacher >> student

# 如果是，说明数据分布有问题
```

#### 解决方案

**方案A：引入负样本**

```python
# 在数据集中加入一些 student > teacher 的样本
# 或者 student ≈ teacher 的样本
# 这样 critic 需要学习更细致的区分
```

**方案B：数据增强**

```python
# 随机交换一部分样本的 teacher 和 student
# 例如，10% 的样本交换标签
def augment_data(batch):
    swap_mask = torch.rand(batch_size) < 0.1
    for i in range(batch_size):
        if swap_mask[i]:
            # 交换 teacher 和 student
            batch["teacher_response"][i], batch["responses"][i] = \
                batch["responses"][i], batch["teacher_response"][i]
    return batch
```

### 原因4：Loss 设计问题 ⭐⭐⭐

#### 问题描述

当前的 discriminator loss 可能**过于关注排序正确性**，而不够关注**分数的校准**。

#### 当前 loss 分析

```python
# 当前的 loss 组成：
d_loss = 1.5 * ranking_loss + score_reg + 0.5 * diff_penalty

# ranking_loss: -log(sigmoid(diff))
# - 只要 teacher > student，loss 就会下降
# - 不关心差距有多大
# - 导致模型倾向于放大差距

# score_reg: 防止分数漂移
# - 权重很小（0.005）
# - 可能不足以约束模型

# diff_penalty: 防止过度自信
# - 只在 diff > 1.5 时生效
# - 阈值可能太高
```

#### 解决方案

**方案A：增加 score_reg 权重**

```python
# 增大正则化权重，防止分数过度分离
score_reg = 0.02 * (teacher_score_raw.pow(2).mean() + student_score_raw.pow(2).mean())
# 从 0.005 增加到 0.02
```

**方案B：降低 diff_penalty 阈值**

```python
# 更早地惩罚过大的差距
diff_penalty = torch.nn.functional.relu(diff - 0.5).pow(2).mean()
# 从 1.5 降低到 0.5
```

**方案C：使用 margin-based loss**

```python
# 只要求 teacher 比 student 好一个 margin，而不是越大越好
target_margin = 0.5
margin_loss = torch.nn.functional.relu(target_margin - diff).mean()
# 当 diff > margin 时，loss = 0
# 当 diff < margin 时，loss > 0，推动增大差距
```

### 原因5：序列级别评分的问题 ⭐⭐⭐⭐

#### 问题描述

你的 critic 使用的是**序列级别的评分**（只在最后一个 token 输出分数），这可能导致：

1. **信息瓶颈**：所有质量信息都压缩到一个标量
2. **训练不稳定**：梯度只通过最后一个 token 回传
3. **过度简化**：模型可能学会简单的二分类（好/坏），而不是细致的质量评估

#### 验证方法

```python
# 检查 critic 的输出分布
# 如果 teacher_value 和 student_value 的分布很分离
# （例如，teacher 都在 [2, 3]，student 都在 [-1, 0]）
# 说明模型学会了简单的二分类
```

#### 解决方案

**方案A：使用 token-level 评分**

```python
# 修改 critic 模型，输出每个 token 的分数
# 然后取平均或加权平均
def _forward_micro_batch(self, micro_batch, compute_teacher=False):
    # ... 前面的代码相同 ...
    
    # 不再只取最后一个 token
    # values = values * last_token_mask
    
    # 而是使用所有 response tokens
    values = values * response_mask  # 所有 response tokens 都有分数
    
    return values
```

**方案B：使用多个评分维度**

```python
# 让 critic 输出多个维度的分数
# 例如：格式分、内容分、流畅度分等
# 然后加权求和
```

### 原因6：训练动态问题 ⭐⭐

#### 问题描述

在 GAD 训练中，critic 和 actor 是**交替训练**的。如果 actor 更新太慢，student 质量提升不明显，critic 就会一直看到质量差距很大的数据。

#### 验证方法

```python
# 观察 actor 的训练指标
actor/policy_loss         # 是否在下降
actor/format_reward_mean  # 是否在提升
actor/kl_divergence       # 是否在合理范围

# 如果 format_reward 长期不提升
# 说明 actor 训练有问题
```

#### 解决方案

1. **增加 actor 训练步数**
2. **调整 actor 学习率**
3. **检查 actor 的 loss 设计**

## 推荐的诊断流程

### Step 1: 检查数据质量差距

```python
# 在训练日志中查看
critic/score_diff          # 期望：< 1.0，实际：？
critic/teacher_value_mean  # 期望：0-2，实际：？
critic/student_value_mean  # 期望：-1-1，实际：？
```

**如果 score_diff > 2.0**：
- 问题：质量差距过大
- 解决：调整 temperature 或 format reward

### Step 2: 检查 loss 组件

```python
# 在训练日志中查看
critic/ranking_loss   # 期望：0.5-0.7，实际：？
critic/score_reg      # 期望：0.01-0.1，实际：？
critic/diff_penalty   # 期望：0-0.1，实际：？
```

**如果 ranking_loss < 0.3**：
- 问题：loss 对差异过于不敏感
- 解决：减小 temperature

**如果 diff_penalty > 0.5**：
- 问题：差距过大，频繁触发惩罚
- 解决：降低阈值或增大 temperature

### Step 3: 检查训练动态

```python
# 观察多个 epoch 的变化
critic/d_acc          # 是否有下降趋势？
critic/score_diff     # 是否在缩小？
actor/format_reward   # 是否在提升？
```

**如果所有指标都不变**：
- 问题：训练陷入局部最优
- 解决：调整学习率或重新初始化

### Step 4: 数据分析

```python
# 手动检查几个样本
# 1. 随机抽取 10 个样本
# 2. 比较 teacher 和 student 的质量
# 3. 评估差距是否真的很大
```

## 立即可以尝试的修复

### 修复1：增大 temperature（最简单）

```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line 1467 附近

# 当前值
temperature = 2.0

# 修改为
temperature = 5.0  # 或 10.0

# 预期效果：d_acc 应该下降到 70-85%
```

### 修复2：降低 diff_penalty 阈值

```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line 1475 附近

# 当前值
diff_penalty = torch.nn.functional.relu(diff - 1.5).pow(2).mean()

# 修改为
diff_penalty = torch.nn.functional.relu(diff - 0.5).pow(2).mean()

# 预期效果：限制 critic 放大差距
```

### 修复3：增大 score_reg 权重

```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line 1472 附近

# 当前值
score_reg = 0.005 * (teacher_score_raw.pow(2).mean() + student_score_raw.pow(2).mean())

# 修改为
score_reg = 0.02 * (teacher_score_raw.pow(2).mean() + student_score_raw.pow(2).mean())

# 预期效果：防止分数漂移到极端值
```

### 修复4：添加监控指标

```python
# 在 dp_critic.py 中添加更多监控
micro_batch_metrics.update({
    # ... 现有指标 ...
    
    # 新增监控
    "critic/score_diff_abs": torch.abs(teacher_score - student_score).mean().item(),
    "critic/teacher_score_std": teacher_score.std().item(),
    "critic/student_score_std": student_score.std().item(),
    "critic/score_overlap": ((teacher_score < student_score.mean()).float().mean() + 
                             (student_score > teacher_score.mean()).float().mean()).item() / 2,
})
```

## 总结

d_acc 持续高位的**最可能原因**是：

1. **Teacher 和 Student 质量差距确实很大**（最可能）
2. **Discriminator loss 的 temperature 太小**（很可能）
3. **数据分布单一**（可能）
4. **序列级别评分的局限性**（可能）

**推荐的修复顺序**：

1. ✅ **立即尝试**：增大 temperature 到 5.0 或 10.0
2. ✅ **同时尝试**：降低 diff_penalty 阈值到 0.5
3. ✅ **同时尝试**：增大 score_reg 权重到 0.02
4. ⏳ **观察效果**：训练 100-200 steps，看 d_acc 是否下降
5. 🔍 **如果无效**：检查数据质量，考虑数据增强或调整 format reward

**预期效果**：

- d_acc 应该下降到 **70-85%** 的合理范围
- score_diff 应该随训练**逐渐缩小**
- student 质量应该**真正提升**
