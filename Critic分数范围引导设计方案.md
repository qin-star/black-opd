# Critic 分数范围引导设计方案

## 问题描述

### 当前问题

1. **分数随机**：Critic 给出的分数没有明确范围，可能是任意值（如 -1.4297）
2. **负分数**：Teacher 和 Student 都可能得到负分数
3. **难以解释**：分数没有明确语义，难以判断 Critic 是否学到正确的评分标准
4. **数值不稳定**：分数范围过大可能导致梯度问题

### 期望目标

1. **Teacher 分数为正**：高质量答案应该得到正分数
2. **合理范围**：分数在可预测的范围内（如 0-10）
3. **清晰语义**：分数有明确含义（0=很差，5=中等，10=优秀）
4. **数值稳定**：避免极端值，提高训练稳定性

## 设计方案

### 方案 1：Score Range Regularization（推荐）

#### 核心思路

通过**软约束**（正则化损失）引导分数到目标范围，而不是硬截断。

#### 目标范围

```python
# Teacher 分数：[5, 10]
# - Teacher 是高质量答案，应该得到较高分数
# - 5 = 中等质量，10 = 优秀质量

# Student 分数：[0, 10]
# - Student 质量不确定，允许更大范围
# - 0 = 很差，5 = 中等，10 = 优秀
```

#### 实现方式

```python
def compute_discriminator_loss(...):
    # 1. 计算原始分数
    teacher_score = teacher_vpreds.sum(dim=-1)
    student_score = student_vpreds.sum(dim=-1)
    
    # 2. Ranking Loss（保持不变）
    diff = teacher_score - student_score
    ranking_loss = -torch.nn.functional.logsigmoid(diff / temperature).mean()
    
    # 3. Score Range Regularization（新增）
    # Teacher 约束：鼓励在 [5, 10] 范围
    teacher_low_penalty = relu(5.0 - teacher_score)   # 惩罚 < 5
    teacher_high_penalty = relu(teacher_score - 10.0) # 惩罚 > 10
    
    # Student 约束：鼓励在 [0, 10] 范围
    student_low_penalty = relu(0.0 - student_score)   # 惩罚 < 0
    student_high_penalty = relu(student_score - 10.0) # 惩罚 > 10
    
    # 总正则化
    score_reg = 0.1 * (
        teacher_low_penalty.pow(2).mean() +
        teacher_high_penalty.pow(2).mean() +
        student_low_penalty.pow(2).mean() +
        student_high_penalty.pow(2).mean()
    )
    
    # 4. 总损失
    d_loss = 3.0 * ranking_loss + score_reg
    
    return d_loss, loss_info
```

#### 工作原理

```python
# 例子 1：Teacher 分数 = 7.0（在目标范围内）
teacher_low_penalty = relu(5.0 - 7.0) = 0  # 无惩罚
teacher_high_penalty = relu(7.0 - 10.0) = 0  # 无惩罚
# 总惩罚 = 0，不影响训练

# 例子 2：Teacher 分数 = 3.0（低于目标）
teacher_low_penalty = relu(5.0 - 3.0) = 2.0  # 有惩罚
penalty = 0.1 * (2.0)^2 = 0.4
# 梯度会推动分数上升

# 例子 3：Teacher 分数 = 12.0（高于目标）
teacher_high_penalty = relu(12.0 - 10.0) = 2.0  # 有惩罚
penalty = 0.1 * (2.0)^2 = 0.4
# 梯度会推动分数下降

# 例子 4：Student 分数 = -1.0（低于目标）
student_low_penalty = relu(0.0 - (-1.0)) = 1.0  # 有惩罚
penalty = 0.1 * (1.0)^2 = 0.1
# 梯度会推动分数上升到 0 以上
```

#### 优势

1. **软约束**：不是硬截断，允许分数暂时超出范围
2. **渐进引导**：通过梯度逐步引导分数到目标范围
3. **不破坏相对关系**：ranking_loss 仍然是主要目标
4. **可调节**：可以调整权重和目标范围

#### 权重选择

```python
# 权重 0.1 的理由
ranking_loss ≈ 0.3-0.7（训练中期）
score_reg ≈ 0.1-0.3（如果分数偏离目标）

# 总损失
d_loss = 3.0 * ranking_loss + score_reg
       = 3.0 * 0.5 + 0.2
       = 1.5 + 0.2
       = 1.7

# score_reg 占比 ≈ 12%
# 足够强以引导分数，但不会压倒 ranking_loss
```

### 方案 2：Target Score Anchoring（备选）

#### 核心思路

为 Teacher 设置一个**目标分数**（如 8.0），通过 MSE loss 引导分数接近目标。

#### 实现方式

```python
def compute_discriminator_loss(...):
    # 1. Ranking Loss（保持不变）
    ranking_loss = -logsigmoid(teacher_score - student_score).mean()
    
    # 2. Target Score Loss（新增）
    teacher_target = 8.0  # Teacher 的目标分数
    target_loss = (teacher_score - teacher_target).pow(2).mean()
    
    # 3. 总损失
    d_loss = 3.0 * ranking_loss + 0.05 * target_loss
    
    return d_loss, loss_info
```

#### 问题

1. **固定目标**：所有 Teacher 都被推向同一个分数（8.0）
2. **破坏区分度**：不同质量的 Teacher 应该有不同分数
3. **与 ranking_loss 冲突**：可能导致训练不稳定

**不推荐**。

### 方案 3：Score Normalization（备选）

#### 核心思路

在前向传播后，将分数归一化到目标范围。

#### 实现方式

```python
def _forward_micro_batch(self, micro_batch, compute_teacher=False):
    # 1. 计算原始分数
    values = self.critic_module(...)
    sequence_value = values.mean()  # 原始分数，可能是任意值
    
    # 2. 归一化到 [0, 10] 范围
    # 使用 sigmoid 映射
    sequence_value_normalized = 10.0 * torch.sigmoid(sequence_value)
    
    # 3. 返回归一化后的分数
    return sequence_value_normalized
```

#### 问题

1. **改变梯度**：归一化会改变梯度流，可能影响训练
2. **固定映射**：sigmoid 的映射是固定的，不够灵活
3. **不解决根本问题**：只是表面上改变分数，模型没有真正学到范围

**不推荐**。

## 推荐方案详细设计

### 方案 1：Score Range Regularization

#### 参数配置

```python
# Teacher 目标范围
TEACHER_SCORE_LOW = 5.0   # 最低期望分数
TEACHER_SCORE_HIGH = 10.0  # 最高期望分数

# Student 目标范围
STUDENT_SCORE_LOW = 0.0    # 最低期望分数
STUDENT_SCORE_HIGH = 10.0   # 最高期望分数

# 正则化权重
SCORE_REG_WEIGHT = 0.1     # 正则化损失的权重
```

#### 分数语义

```python
# Teacher 分数语义
10.0: 完美答案（准确、完整、简洁）
8.0:  优秀答案（准确、完整）
6.0:  良好答案（准确但不够完整）
5.0:  及格答案（基本准确）
< 5.0: 不应该出现（Teacher 应该是高质量的）

# Student 分数语义
10.0: 完美答案（达到 Teacher 水平）
8.0:  优秀答案
5.0:  中等答案
2.0:  较差答案
0.0:  很差答案（完全错误或无意义）
< 0.0: 不应该出现
```

#### 训练动态

```python
# 训练初期（Step 0-50）
Teacher: -1.4 → 惩罚 = 0.1 * (5.0 - (-1.4))^2 = 4.1
Student: -2.5 → 惩罚 = 0.1 * (0.0 - (-2.5))^2 = 0.6
# 梯度推动分数上升

# 训练中期（Step 50-150）
Teacher: 3.0 → 惩罚 = 0.1 * (5.0 - 3.0)^2 = 0.4
Student: 1.0 → 惩罚 = 0（在范围内）
# Teacher 继续上升，Student 稳定

# 训练后期（Step 150+）
Teacher: 7.0 → 惩罚 = 0（在范围内）
Student: 4.0 → 惩罚 = 0（在范围内）
# 分数稳定在目标范围
```

#### 监控指标

```python
# 新增监控指标
"critic/teacher_score_mean": 7.0      # 应该在 5-10 范围
"critic/student_score_mean": 4.0      # 应该在 0-10 范围
"critic/score_reg": 0.05              # 应该逐渐减小到接近 0
"critic/teacher_in_range_ratio": 0.95 # Teacher 在范围内的比例
"critic/student_in_range_ratio": 0.98 # Student 在范围内的比例
```

## 风险与缓解

### 风险 1：需要重新 Warmup

**风险**：当前 Critic 已经学到负分数，修改后需要重新训练

**缓解**：
- 这是必要的，当前分数确实不合理
- Warmup 相对较快（约 12 小时）
- 长期收益大于短期成本

### 风险 2：可能影响区分度

**风险**：约束分数范围可能降低 Critic 的区分能力

**缓解**：
- 使用软约束，不是硬截断
- ranking_loss 仍然是主要目标（权重 3.0）
- score_reg 只是辅助（权重 0.1）
- 10 分的范围足够区分不同质量

### 风险 3：超参数需要调整

**风险**：权重 0.1 可能不是最优的

**缓解**：
- 从 0.1 开始，观察前 50 步
- 如果分数上升太慢，增大到 0.2
- 如果分数震荡，减小到 0.05
- 监控 score_reg 的值，应该在 0.1-0.5 范围

### 风险 4：与现有训练冲突

**风险**：新的正则化可能与 ranking_loss 冲突

**缓解**：
- ranking_loss 权重更大（3.0 vs 0.1）
- 两者目标一致（都希望 teacher > student）
- score_reg 只是约束范围，不改变相对关系

## 实施计划

### 阶段 1：代码修改（✅ 已完成）

1. ✅ 修改 `compute_discriminator_loss` 函数
2. ✅ 添加 Score Range Regularization
3. ✅ 添加监控指标

**实施细节**：
- 文件：`verl/verl/trainer/ppo/core_algos.py`
- 修改位置：`compute_discriminator_loss` 函数
- 替换了原有的 Score Drift Regularization
- 新增了 6 个监控指标

### 阶段 2：重新 Warmup（⏳ 待执行 - 约 12 小时）

1. 清理旧的 checkpoint
2. 重新运行 Warmup 脚本
3. 观察前 50 步的分数变化

### 阶段 3：验证效果（1 小时）

1. 检查 `critic/teacher_score_mean`（应该在 5-10）
2. 检查 `critic/student_score_mean`（应该在 0-10）
3. 检查 `critic/score_diff`（应该 > 0）
4. 检查 `critic/d_acc`（应该在 0.5-0.7）

### 阶段 4：调整超参数（可选）

如果效果不理想：
- 调整 `SCORE_REG_WEIGHT`（0.05-0.2）
- 调整目标范围（如 Teacher [6, 10]）
- 调整 ranking_loss 权重（2.0-5.0）

## 预期效果

### 修改前

```python
Teacher: -1.4297  # 负数，无明确语义
Student: -2.5000  # 负数，无明确语义
Score Diff: 1.07  # 相对关系正确，但绝对值随机
```

### 修改后

```python
Teacher: 7.5  # 正数，语义清晰（优秀答案）
Student: 4.2  # 正数，语义清晰（中等答案）
Score Diff: 3.3  # 相对关系正确，绝对值合理
```

### 训练曲线

```python
# Teacher 分数
Step 0:   -1.4 → 惩罚大，快速上升
Step 50:   3.0 → 惩罚中，继续上升
Step 100:  5.5 → 惩罚小，缓慢上升
Step 150:  7.0 → 无惩罚，稳定
Step 200+: 7.0-8.0 → 稳定在目标范围

# Student 分数
Step 0:   -2.5 → 惩罚大，快速上升
Step 50:   1.0 → 惩罚小，继续上升
Step 100:  3.0 → 无惩罚，稳定
Step 150+: 3.0-5.0 → 稳定在目标范围
```

## Temperature 参数的影响分析

### 当前 Temperature 机制

```python
# 当前代码（自适应 temperature）
if adaptive_temperature:
    current_diff_abs = diff.abs().mean().item()
    # 映射函数：
    # - diff=0.05 → T=0.335 (强梯度，快速学习)
    # - diff=0.5  → T=0.65  (中等梯度)
    # - diff=1.0  → T=1.0   (标准梯度)
    # - diff=2.0  → T=1.7   (弱梯度，精细区分)
    adaptive_temp = torch.clamp(
        torch.tensor(0.3 + current_diff_abs * 0.7, device=diff.device),
        min=0.3,
        max=2.0
    )
    temperature = adaptive_temp.item()

scaled_diff = diff / temperature
ranking_loss = -torch.nn.functional.logsigmoid(scaled_diff).mean()
```

### 使用 Score Range Regularization 后的影响

#### 分数差异的变化

```python
# 修改前（无范围约束）
Teacher: -1.4, -2.5, 3.0, -5.0  # 随机分布
Student: -2.5, -3.0, 1.0, -6.0
Diff:     1.1,  0.5, 2.0,  1.0  # 差异不稳定

# 修改后（有范围约束）
Teacher: 7.0, 8.0, 6.5, 7.5  # 集中在 [5, 10]
Student: 4.0, 5.0, 3.0, 4.5  # 集中在 [0, 10]
Diff:    3.0, 3.0, 3.5, 3.0  # 差异更稳定
```

#### Temperature 的适配

**关键发现**：Score Range Regularization 会使分数差异**更稳定、更可预测**。

```python
# 修改前：diff 范围很大
diff 范围: [-5, +10]  # 不可预测
平均 diff: 1.0-2.0   # 不稳定

# 修改后：diff 范围收敛
diff 范围: [1.0, 5.0]  # 可预测
平均 diff: 2.5-3.5    # 稳定

# 自适应 temperature 的行为
训练初期: diff ≈ 1.0 → T ≈ 1.0
训练中期: diff ≈ 2.5 → T ≈ 2.05
训练后期: diff ≈ 3.0 → T ≈ 2.4（接近上限 2.0）
```

### 建议的 Temperature 调整

#### 方案 A：保持自适应，调整范围（推荐）

```python
# 调整自适应 temperature 的映射范围
if adaptive_temperature:
    current_diff_abs = diff.abs().mean().item()
    # 新映射函数（适配更大的 diff）：
    # - diff=1.0 → T=0.5  (强梯度)
    # - diff=2.0 → T=1.0  (中等梯度)
    # - diff=3.0 → T=1.5  (标准梯度)
    # - diff=5.0 → T=2.5  (弱梯度)
    adaptive_temp = torch.clamp(
        torch.tensor(0.5 * current_diff_abs, device=diff.device),
        min=0.5,   # 提高下限（从 0.3 到 0.5）
        max=3.0    # 提高上限（从 2.0 到 3.0）
    )
    temperature = adaptive_temp.item()
```

**理由**：
1. Score Range Regularization 会使 diff 增大（从 1.0 增大到 3.0）
2. 需要更大的 temperature 来平衡梯度强度
3. 保持自适应机制，自动适应训练阶段

#### 方案 B：使用固定 temperature（备选）

```python
# 简化为固定 temperature
temperature = 1.5  # 固定值

scaled_diff = diff / temperature
ranking_loss = -torch.nn.functional.logsigmoid(scaled_diff).mean()
```

**理由**：
1. Score Range Regularization 使 diff 更稳定
2. 不再需要复杂的自适应机制
3. 更简单，更容易调试

**缺点**：
- 失去了自适应能力
- 需要手动调整

#### 方案 C：关闭自适应，使用当前固定值（最简单）

```python
# 使用当前配置的固定 temperature
# 在训练脚本中：adaptive_temperature=False, temperature=1.0
temperature = 1.0  # 或者从配置读取

scaled_diff = diff / temperature
ranking_loss = -torch.nn.functional.logsigmoid(scaled_diff).mean()
```

**理由**：
- 不需要修改代码
- 只需要在训练脚本中设置 `adaptive_temperature=False`

### Temperature 与 Score Range 的协同效应

#### 训练动态分析

```python
# 训练初期（Step 0-50）
Teacher: -1.4 → 5.0（被 score_reg 推高）
Student: -2.5 → 1.0（被 score_reg 推高）
Diff: 1.1 → 4.0（增大）

# Temperature 的作用
T = 1.0（固定）或 T = 2.0（自适应）
scaled_diff = 4.0 / 2.0 = 2.0
ranking_loss = -log(sigmoid(2.0)) = 0.13（较小，梯度弱）

# 效果：score_reg 主导训练，快速建立分数范围

# 训练中期（Step 50-150）
Teacher: 5.0 → 7.0（继续上升）
Student: 1.0 → 3.0（继续上升）
Diff: 4.0 → 4.0（稳定）

# Temperature 的作用
T = 2.0（自适应）
scaled_diff = 4.0 / 2.0 = 2.0
ranking_loss = 0.13（稳定）

# 效果：ranking_loss 和 score_reg 共同作用，精细调整

# 训练后期（Step 150+）
Teacher: 7.0 → 7.5（微调）
Student: 3.0 → 3.5（微调）
Diff: 4.0 → 4.0（稳定）

# Temperature 的作用
T = 2.0（自适应）
scaled_diff = 4.0 / 2.0 = 2.0
ranking_loss = 0.13（稳定）

# 效果：分数稳定在目标范围，训练收敛
```

#### 梯度强度对比

```python
# 场景 1：diff = 1.0, T = 1.0
scaled_diff = 1.0
sigmoid(1.0) = 0.73
ranking_loss = -log(0.73) = 0.31
梯度强度：中等

# 场景 2：diff = 3.0, T = 1.0
scaled_diff = 3.0
sigmoid(3.0) = 0.95
ranking_loss = -log(0.95) = 0.05
梯度强度：很弱（可能导致训练停滞）

# 场景 3：diff = 3.0, T = 2.0
scaled_diff = 1.5
sigmoid(1.5) = 0.82
ranking_loss = -log(0.82) = 0.20
梯度强度：适中（推荐）

# 场景 4：diff = 3.0, T = 3.0
scaled_diff = 1.0
sigmoid(1.0) = 0.73
ranking_loss = -log(0.73) = 0.31
梯度强度：中等（也可以）
```

### 推荐配置

#### 配置 1：自适应 temperature + 调整范围（最推荐）

```python
# 在 compute_discriminator_loss 中
if adaptive_temperature:
    current_diff_abs = diff.abs().mean().item()
    # 新映射：适配更大的 diff
    adaptive_temp = torch.clamp(
        torch.tensor(0.5 * current_diff_abs, device=diff.device),
        min=0.5,   # 从 0.3 提高到 0.5
        max=3.0    # 从 2.0 提高到 3.0
    )
    temperature = adaptive_temp.item()
```

**优势**：
- 自动适应分数差异的变化
- 训练初期（diff 小）用小 T，快速学习
- 训练后期（diff 大）用大 T，精细调整

#### 配置 2：固定 temperature = 2.0（备选）

```python
# 在训练脚本中
adaptive_temperature=False
temperature=2.0
```

**优势**：
- 简单，易于调试
- 适合 diff 稳定在 3.0 左右的情况

#### 配置 3：保持当前配置（最简单）

```python
# 不修改 temperature 相关代码
# 使用当前的自适应机制
adaptive_temperature=True
temperature=1.0  # 初始值
```

**优势**：
- 不需要修改代码
- 当前机制可能已经足够好

### 实验建议

#### 阶段 1：使用当前配置（Step 0-50）

```python
# 不修改 temperature
adaptive_temperature=True
temperature=1.0
```

**观察指标**：
- `critic/temperature`：应该在 1.0-2.0 范围
- `critic/score_diff`：应该从 1.0 增大到 3.0
- `critic/ranking_loss`：应该稳定下降

#### 阶段 2：根据观察调整（Step 50+）

**如果 ranking_loss 下降太慢**：
```python
# 增大 temperature 上限
max_temperature = 3.0  # 从 2.0 增大到 3.0
```

**如果 score_diff 震荡**：
```python
# 减小 temperature
temperature = 1.5  # 固定值
adaptive_temperature = False
```

**如果训练稳定**：
```python
# 保持当前配置
# 不需要修改
```

### 监控指标

```python
# 新增监控指标
"critic/temperature": 1.5              # 当前使用的 temperature
"critic/scaled_diff": 2.0              # diff / temperature
"critic/ranking_loss": 0.20            # 应该在 0.1-0.3 范围
"critic/score_diff": 3.0               # 应该在 2.0-4.0 范围
"critic/teacher_score_mean": 7.0       # 应该在 5-10 范围
"critic/student_score_mean": 4.0       # 应该在 0-10 范围
```

### 决策树

```
是否需要修改 temperature？
│
├─ 如果想要最大灵活性
│  └─ 使用配置 1：自适应 + 调整范围
│     - 修改映射函数
│     - min=0.5, max=3.0
│
├─ 如果想要简单稳定
│  └─ 使用配置 2：固定 temperature=2.0
│     - adaptive_temperature=False
│     - temperature=2.0
│
└─ 如果想要最小改动
   └─ 使用配置 3：保持当前配置
      - 不修改代码
      - 观察训练效果
      - 必要时再调整
```

## 总结

### 推荐方案

**方案 1：Score Range Regularization**

### 核心优势

1. **软约束**：渐进引导，不破坏训练
2. **清晰语义**：分数有明确含义
3. **数值稳定**：避免极端值
4. **易于监控**：可以直观判断训练效果

### 关键参数

```python
# Score Range Regularization
TEACHER_SCORE_LOW = 5.0
TEACHER_SCORE_HIGH = 10.0
STUDENT_SCORE_LOW = 0.0
STUDENT_SCORE_HIGH = 10.0
SCORE_REG_WEIGHT = 0.1

# Temperature（推荐配置 1）
ADAPTIVE_TEMPERATURE = True
TEMPERATURE_MIN = 0.5  # 从 0.3 提高
TEMPERATURE_MAX = 3.0  # 从 2.0 提高
TEMPERATURE_MAPPING = 0.5 * diff  # 新映射函数
```

### Temperature 修改建议

**推荐**：使用配置 1（自适应 + 调整范围）
- 修改映射函数以适配更大的 diff
- 提高 min 和 max 范围

**备选**：使用配置 3（保持当前配置）
- 先不修改，观察效果
- 如果训练稳定，不需要改动

### 下一步

1. ✅ 审核设计方案
2. ⏳ 确认 temperature 配置（配置 1 或配置 3）
3. ⏳ 实施代码修改
4. ⏳ 重新 Warmup 训练
5. ⏳ 观察前 50 步，必要时调整

---

**创建日期**: 2026-02-05
**作者**: AI Assistant
**状态**: ✅ 已实施，等待训练验证
**最后更新**: 2026-02-05（实施完成）

## 实施记录

### 2026-02-05 - 代码实施完成

**修改文件**: `verl/verl/trainer/ppo/core_algos.py`

**修改内容**:
1. ✅ 将 Score Drift Regularization 替换为 Score Range Regularization
2. ✅ 添加 Teacher [5, 10] 和 Student [0, 10] 的范围约束
3. ✅ 新增 6 个监控指标
4. ✅ 保持 Temperature 配置不变

**详细文档**:
- `Critic指标优化完成说明.md` - 完整实施说明
- `Critic分数范围优化快速参考.md` - 快速参考指南

**下一步**: 重新运行 Warmup 训练（约 12 小时）
