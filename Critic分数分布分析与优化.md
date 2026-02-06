# Critic 分数分布分析与优化

## 当前分数分布

### 观察到的数据

```
全局统计 (64 个样本):
  Teacher 平均分:  6.4688
  Student 平均分: -6.5000
  平均分差:       13.0000
  Teacher > Student: 93.8%

分数范围:
  Student: [-13.6875, 16.6250]  ← 范围 30.3125
  Teacher: [-4.8438, 16.6250]   ← 范围 21.4688
```

### 关键问题

1. **分数范围过大**
   - Student 分数范围：30.3
   - Teacher 分数范围：21.5
   - 这在未训练的基础模型中是正常的，但需要关注

2. **平均分差很大**
   - 平均分差：13.0
   - 这说明 Critic 已经能够区分 teacher 和 student
   - 但分差可能过大

3. **Teacher > Student 比例很高**
   - 93.8% 的样本 teacher 分数更高
   - 理想情况应该在 60-80%
   - 可能存在过度区分的问题

## 问题分析

### 问题 1：分数范围过大

#### 原因分析

**A. 未训练的基础模型**
```
Qwen3-14B 基础模型的初始权重：
- 随机初始化或预训练权重
- 输出层的 logits 可能有较大方差
- 没有经过 value head 的训练
```

**B. Last Token 的特殊性**
```
Last token 的 value 可能：
- 受到整个序列的影响
- 在不同样本间方差较大
- 没有经过归一化
```

**C. 序列长度差异**
```
不同样本的 response 长度不同：
- 短回答：1-5 tokens
- 长回答：50-200 tokens
- 长度可能影响 value 的大小
```

#### 潜在风险

**风险 1：梯度爆炸/消失**
```python
# 如果分数范围过大
teacher_score = 16.6
student_score = -13.7
diff = 30.3

# Sigmoid 函数
sigmoid(diff / temperature) ≈ 1.0  # 饱和
gradient ≈ 0  # 梯度消失

# 或者
sigmoid(diff / temperature) ≈ 0.0  # 饱和
gradient ≈ 0  # 梯度消失
```

**风险 2：训练不稳定**
```python
# 分数波动大
step 1: score_diff = 13.0
step 2: score_diff = 25.0
step 3: score_diff = 5.0

# 导致
- 梯度不稳定
- 学习率难以调整
- 训练震荡
```

**风险 3：Reward 发散**
```python
# 在 GAD 阶段
actor_reward = critic_score

# 如果 critic_score 范围过大
actor_reward ∈ [-13.7, 16.6]

# 可能导致
- Actor 的 advantage 计算不准确
- Policy gradient 方差过大
- Actor 训练不稳定
```

### 问题 2：平均分差过大

#### 当前状态

```
平均分差: 13.0
Teacher > Student: 93.8%
```

**分析**：
- 分差 13.0 说明 Critic 能够强烈区分 teacher 和 student
- 93.8% 的准确率说明区分能力很强
- 但这可能是**过度区分**

#### 理想状态

```
平均分差: 1-5
Teacher > Student: 60-80%
```

**理由**：
- 分差太小（< 1）：Critic 区分能力不足
- 分差适中（1-5）：Critic 有区分能力，但不过度
- 分差太大（> 10）：Critic 过度自信，可能过拟合

#### 过度区分的风险

**风险 1：Critic 过拟合**
```
Critic 学习到的模式：
- "所有 teacher 都是好的" → 高分
- "所有 student 都是差的" → 低分

问题：
- 没有学到细微的质量差异
- 无法给出有区分度的奖励
- Actor 收到的信号过于粗糙
```

**风险 2：Actor 训练困难**
```
Actor 收到的 reward：
- 好的 student response: -6.5
- 差的 student response: -13.7
- Teacher response: 6.5

问题：
- 所有 student responses 都是负奖励
- Actor 难以学习哪些行为更好
- 可能导致 Actor 崩溃
```

## 解决方案

### 方案 1：分数归一化（推荐）

**目标**：将分数范围限制在合理区间

#### 实现方式 A：Batch 归一化

```python
# 在 discriminator loss 计算中
def compute_discriminator_loss_with_normalization(
    student_vpreds, teacher_vpreds,
    response_mask, teacher_response_mask,
    temperature=0.5
):
    # 计算原始分数
    student_score = (student_vpreds * response_mask).sum(dim=-1)
    teacher_score = (teacher_vpreds * teacher_response_mask).sum(dim=-1)
    
    # Batch 归一化
    all_scores = torch.cat([student_score, teacher_score])
    mean = all_scores.mean()
    std = all_scores.std() + 1e-8
    
    student_score_norm = (student_score - mean) / std
    teacher_score_norm = (teacher_score - mean) / std
    
    # 计算 loss
    diff = teacher_score_norm - student_score_norm
    ranking_loss = -torch.nn.functional.logsigmoid(diff / temperature).mean()
    
    return ranking_loss
```

**优点**：
- 自动适应不同的分数范围
- 保持相对顺序
- 稳定训练

**缺点**：
- 依赖 batch 统计
- 不同 batch 间可能不一致

#### 实现方式 B：Tanh 归一化

```python
# 在 forward 后应用
def normalize_scores(scores, scale=5.0):
    """
    使用 tanh 将分数归一化到 [-scale, scale]
    """
    return scale * torch.tanh(scores / scale)

# 在 _forward_micro_batch 中
values = self._forward_micro_batch(...)
values = normalize_scores(values, scale=5.0)  # 限制在 [-5, 5]
```

**优点**：
- 平滑的非线性变换
- 自动限制范围
- 保持单调性

**缺点**：
- 可能压缩极端值
- 需要调整 scale 参数

#### 实现方式 C：Clipping

```python
# 简单的裁剪
def clip_scores(scores, min_val=-10.0, max_val=10.0):
    return torch.clamp(scores, min=min_val, max=max_val)

# 在 _forward_micro_batch 中
values = self._forward_micro_batch(...)
values = clip_scores(values, min_val=-10.0, max_val=10.0)
```

**优点**：
- 简单直接
- 防止极端值

**缺点**：
- 硬截断，可能丢失信息
- 不够平滑

### 方案 2：调整 Temperature（推荐）

**目标**：通过增大 temperature 来减小分数差异的影响

```python
# 当前 temperature
temperature = 0.5

# 增大 temperature
temperature = 2.0  # 或更大

# 效果
diff = 13.0
sigmoid(diff / 0.5) = sigmoid(26.0) ≈ 1.0  # 饱和
sigmoid(diff / 2.0) = sigmoid(6.5) ≈ 0.998  # 接近饱和
sigmoid(diff / 5.0) = sigmoid(2.6) ≈ 0.93   # 不饱和
```

**建议**：
- 当前 temperature = 0.5
- 建议增大到 1.0-2.0
- 观察训练稳定性

### 方案 3：Score Regularization（已实施）

**当前实现**：
```python
# 在 core_algos.py 中
score_threshold = 5.0
teacher_extreme = torch.nn.functional.relu(teacher_score.abs() - score_threshold)
student_extreme = torch.nn.functional.relu(student_score.abs() - score_threshold)
score_reg = 0.001 * (teacher_extreme.pow(2).mean() + student_extreme.pow(2).mean())
```

**优化建议**：
```python
# 增大正则化权重
score_reg_weight = 0.01  # 从 0.001 增加到 0.01

# 或者降低阈值
score_threshold = 3.0  # 从 5.0 降低到 3.0
```

### 方案 4：使用 Layer Normalization

**目标**：在 Critic 模型中添加 Layer Norm

```python
class CriticWithLayerNorm(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.layer_norm = nn.LayerNorm(base_model.config.hidden_size)
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, position_ids):
        outputs = self.base_model(input_ids, attention_mask, position_ids)
        hidden_states = outputs.last_hidden_state
        
        # Layer Normalization
        normalized = self.layer_norm(hidden_states)
        
        # Value head
        values = self.value_head(normalized)
        return values
```

**优点**：
- 稳定训练
- 减小方差
- 改善梯度流

**缺点**：
- 需要修改模型架构
- 需要重新训练

## 推荐实施方案

### 短期（立即实施）

#### 1. 增大 Temperature

```python
# 在训练脚本或配置中
temperature = 2.0  # 从 0.5 增加到 2.0
```

**修改位置**：
- `verl/verl/workers/critic/dp_critic.py`
- `update_critic` 方法中调用 `compute_discriminator_loss` 的地方

```python
d_loss, loss_info = core_algos.compute_discriminator_loss(
    student_vpreds=student_vpreds,
    teacher_vpreds=teacher_vpreds,
    response_mask=response_mask,
    teacher_response_mask=teacher_response_mask,
    temperature=2.0,  # 增大 temperature
    adaptive_temperature=False,  # 暂时关闭自适应
)
```

#### 2. 增强 Score Regularization

```python
# 在 core_algos.py 中
score_threshold = 3.0  # 从 5.0 降低
score_reg_weight = 0.01  # 从 0.001 增加

score_reg = score_reg_weight * (teacher_extreme.pow(2).mean() + student_extreme.pow(2).mean())
```

### 中期（1-2 天）

#### 3. 添加 Batch Normalization

```python
# 在 compute_discriminator_loss 中
def compute_discriminator_loss_with_batch_norm(
    student_vpreds, teacher_vpreds,
    response_mask, teacher_response_mask,
    temperature=2.0
):
    # 计算分数
    student_score = (student_vpreds * response_mask).sum(dim=-1)
    teacher_score = (teacher_vpreds * teacher_response_mask).sum(dim=-1)
    
    # Batch 归一化（可选）
    if use_batch_norm:
        all_scores = torch.cat([student_score, teacher_score])
        mean = all_scores.mean()
        std = all_scores.std() + 1e-8
        student_score = (student_score - mean) / std
        teacher_score = (teacher_score - mean) / std
    
    # 计算 loss
    diff = teacher_score - student_score
    ranking_loss = -torch.nn.functional.logsigmoid(diff / temperature).mean()
    
    return ranking_loss
```

### 长期（可选）

#### 4. 改用平均值而非 Last Token

```python
# 在 _forward_micro_batch 中
# 不使用 last token mask
values_mean = (values * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
return values_mean
```

**优点**：
- 更稳定的分数
- 对长度不敏感
- 减小方差

## 监控指标

### 关键指标

1. **分数范围**
   ```
   当前: Student [-13.7, 16.6], Teacher [-4.8, 16.6]
   目标: Student [-10, 10], Teacher [-10, 10]
   ```

2. **平均分差**
   ```
   当前: 13.0
   目标: 1-5
   ```

3. **Teacher > Student 比例**
   ```
   当前: 93.8%
   目标: 60-80%
   ```

4. **训练稳定性**
   ```
   观察:
   - Loss 曲线是否平滑
   - 梯度范数是否稳定
   - d_acc 是否在合理范围
   ```

### 警告阈值

```python
# 分数范围过大
if student_score.max() - student_score.min() > 30:
    logger.warning("Score range too large!")

# 平均分差过大
if abs(teacher_score.mean() - student_score.mean()) > 10:
    logger.warning("Average score difference too large!")

# 准确率过高
if d_acc > 0.95:
    logger.warning("Discriminator accuracy too high, may overfit!")
```

## 总结

### 当前问题

1. ⚠️ **分数范围过大**（30.3）
2. ⚠️ **平均分差过大**（13.0）
3. ⚠️ **Teacher > Student 比例过高**（93.8%）

### 潜在风险

1. 梯度饱和/消失
2. 训练不稳定
3. Reward 发散
4. Critic 过拟合
5. Actor 训练困难

### 推荐方案

**立即实施**：
1. ✅ 增大 temperature（0.5 → 2.0）
2. ✅ 增强 score regularization

**中期实施**：
3. ⚠️ 添加 batch normalization（可选）

**长期考虑**：
4. ⚖️ 改用平均值而非 last token（可选）

### 预期效果

```
修复前:
  分数范围: 30.3
  平均分差: 13.0
  准确率: 93.8%

修复后:
  分数范围: < 20
  平均分差: 3-5
  准确率: 70-80%
```

**建议先实施方案 1 和 2，观察效果后再决定是否需要更激进的方案。**
