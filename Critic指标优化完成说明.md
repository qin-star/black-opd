# Critic 分数范围引导实施完成说明

## 实施日期
2026-02-05

## 实施内容

### 1. 核心修改

**文件**: `verl/verl/trainer/ppo/core_algos.py`

**函数**: `compute_discriminator_loss`

**修改内容**: 
1. 将原有的 Score Drift Regularization 替换为 Score Range Regularization
2. 调整 adaptive temperature 的映射范围以适配更大的 score_diff

#### 修改前（Score Drift Regularization）
```python
# Component 2: Score Drift Regularization
# 目标：防止判别器输出值整体飘向正负无穷
score_threshold = 3.0
teacher_extreme = torch.nn.functional.relu(teacher_score_raw.abs() - score_threshold)
student_extreme = torch.nn.functional.relu(student_score_raw.abs() - score_threshold)
score_reg = 0.01 * (teacher_extreme.pow(2).mean() + student_extreme.pow(2).mean())
```

**问题**：
- 只约束极端值（|score| > 3.0），不引导分数到特定范围
- 无法解决负分数问题
- 分数没有明确语义

#### 修改后（Score Range Regularization）
```python
# Component 2: Score Range Regularization
# 目标：引导 Critic 给出有明确语义的分数范围

# Teacher 约束：鼓励在 [5, 10] 范围
TEACHER_SCORE_LOW = 5.0
TEACHER_SCORE_HIGH = 10.0
teacher_low_penalty = torch.nn.functional.relu(TEACHER_SCORE_LOW - teacher_score_raw)
teacher_high_penalty = torch.nn.functional.relu(teacher_score_raw - TEACHER_SCORE_HIGH)

# Student 约束：鼓励在 [0, 10] 范围
STUDENT_SCORE_LOW = 0.0
STUDENT_SCORE_HIGH = 10.0
student_low_penalty = torch.nn.functional.relu(STUDENT_SCORE_LOW - student_score_raw)
student_high_penalty = torch.nn.functional.relu(student_score_raw - STUDENT_SCORE_HIGH)

# 总正则化损失
SCORE_REG_WEIGHT = 0.1
score_reg = SCORE_REG_WEIGHT * (
    teacher_low_penalty.pow(2).mean() +
    teacher_high_penalty.pow(2).mean() +
    student_low_penalty.pow(2).mean() +
    student_high_penalty.pow(2).mean()
)
```

**优势**：
- ✅ 软约束：渐进引导，不破坏训练
- ✅ 明确范围：Teacher [5, 10], Student [0, 10]
- ✅ 清晰语义：分数有明确含义
- ✅ 解决负分数：通过惩罚 < 0 的分数

#### 修改 2：Adaptive Temperature 映射调整

**修改前**：
```python
# 旧映射函数
adaptive_temp = torch.clamp(
    torch.tensor(0.3 + current_diff_abs * 0.7, device=diff.device),
    min=0.3,
    max=2.0
)
# diff=3.0 → T=2.4 → clamp 到 2.0（达到上限，失去自适应能力）
```

**问题**：
- Score Range Regularization 会使 diff 增大到 3.0-4.0
- 旧映射会使 temperature 卡在上限 2.0
- 失去自适应能力

**修改后**：
```python
# 新映射函数（适配更大的 diff）
adaptive_temp = torch.clamp(
    torch.tensor(0.5 * current_diff_abs, device=diff.device),
    min=0.5,   # 从 0.3 提高到 0.5
    max=3.0    # 从 2.0 提高到 3.0
)
# diff=1.0 → T=0.5  (强梯度)
# diff=2.0 → T=1.0  (中等梯度)
# diff=3.0 → T=1.5  (标准梯度)
# diff=4.0 → T=2.0  (弱梯度)
# diff=5.0 → T=2.5  (很弱梯度)
```

**优势**：
- ✅ 适配更大的 diff 范围（3.0-4.0）
- ✅ 保持自适应能力（不会卡在上限）
- ✅ 更简洁的映射函数（0.5 * diff）
- ✅ 自动平衡梯度强度

### 2. 新增监控指标

在 `loss_info` 字典中新增了 6 个监控指标：

```python
# 新增：Score Range Regularization 监控指标
"teacher_in_range_ratio": ((teacher_score_raw >= TEACHER_SCORE_LOW) & 
                           (teacher_score_raw <= TEACHER_SCORE_HIGH)).float().mean().detach().item(),
"student_in_range_ratio": ((student_score_raw >= STUDENT_SCORE_LOW) & 
                           (student_score_raw <= STUDENT_SCORE_HIGH)).float().mean().detach().item(),
"teacher_score_min": teacher_score_raw.min().detach().item(),
"teacher_score_max": teacher_score_raw.max().detach().item(),
"student_score_min": student_score_raw.min().detach().item(),
"student_score_max": student_score_raw.max().detach().item(),
```

**指标说明**：
- `teacher_in_range_ratio`: Teacher 分数在 [5, 10] 范围内的比例（目标：> 0.9）
- `student_in_range_ratio`: Student 分数在 [0, 10] 范围内的比例（目标：> 0.9）
- `teacher_score_min/max`: Teacher 分数的最小/最大值（观察分布）
- `student_score_min/max`: Student 分数的最小/最大值（观察分布）

## 设计参数

### 目标范围
```python
# Teacher 分数：[5, 10]
# - Teacher 是高质量答案，应该得到较高分数
# - 5 = 及格质量，10 = 完美质量

# Student 分数：[0, 10]
# - Student 质量不确定，允许更大范围
# - 0 = 很差，5 = 中等，10 = 优秀
```

### 分数语义
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

### 权重配置
```python
SCORE_REG_WEIGHT = 0.1  # 正则化权重

# 总损失
d_loss = 3.0 * ranking_loss + score_reg

# 权重比例
# ranking_loss 权重: 3.0 (主要目标)
# score_reg 权重: 0.1 (辅助约束)
# 比例: 30:1
```

**权重选择理由**：
- `ranking_loss` 仍然是主要目标（权重 3.0）
- `score_reg` 只是辅助约束（权重 0.1）
- 足够强以引导分数，但不会压倒 ranking_loss
- 预期 score_reg 占总损失的 10-15%

## 工作原理

### 训练动态

```python
# 训练初期（Step 0-50）
Teacher: -1.4 → 5.0（被 score_reg 推高）
Student: -2.5 → 1.0（被 score_reg 推高）
Diff: 1.1 → 4.0（增大）

# 训练中期（Step 50-150）
Teacher: 5.0 → 7.0（继续上升）
Student: 1.0 → 3.0（继续上升）
Diff: 4.0 → 4.0（稳定）

# 训练后期（Step 150+）
Teacher: 7.0 → 7.5（微调）
Student: 3.0 → 3.5（微调）
Diff: 4.0 → 4.0（稳定）
```

### 惩罚机制示例

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

## Temperature 配置

### 已修改（✅ 完成）

**修改内容**: 调整 adaptive temperature 的映射范围

**修改前**:
```python
# 旧映射：0.3 + diff * 0.7, 范围 [0.3, 2.0]
# 问题：diff=3.0 时会卡在上限 2.0
```

**修改后**:
```python
# 新映射：0.5 * diff, 范围 [0.5, 3.0]
# 优势：适配更大的 diff (3.0-4.0)，保持自适应能力
```

### 映射对比

| diff | 旧 T (0.3+0.7*diff) | 新 T (0.5*diff) | 说明 |
|------|---------------------|-----------------|------|
| 1.0  | 1.0                 | 0.5             | 训练初期，强梯度 |
| 2.0  | 1.7                 | 1.0             | 训练中期，中等梯度 |
| 3.0  | 2.0 (上限)          | 1.5             | 训练后期，标准梯度 |
| 4.0  | 2.0 (上限)          | 2.0             | 稳定阶段，弱梯度 |
| 5.0  | 2.0 (上限)          | 2.5             | 精细调整，很弱梯度 |

**关键改进**：
- ✅ 避免 temperature 卡在上限
- ✅ 保持自适应能力
- ✅ 更简洁的映射函数

## 预期效果

### 修改前
```python
Teacher: -1.4297  # 负数，无明确语义
Student: -2.5000  # 负数，无明确语义
Score Diff: 1.07  # 相对关系正确，但绝对值随机
```

### 修改后（预期）
```python
Teacher: 7.5  # 正数，语义清晰（优秀答案）
Student: 4.2  # 正数，语义清晰（中等答案）
Score Diff: 3.3  # 相对关系正确，绝对值合理
```

### 训练曲线（预期）
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

# score_reg
Step 0:   4.0 → 很大（分数远离目标）
Step 50:  0.8 → 减小（分数接近目标）
Step 100: 0.2 → 很小（分数进入目标范围）
Step 150+: 0.05 → 接近 0（分数稳定在目标范围）
```

## 下一步操作

### 1. 重新 Warmup 训练（必须）

**原因**：
- 当前 Critic 已经学到负分数
- 修改后需要重新训练以学习新的分数范围

**操作**：
```bash
# 清理旧的 checkpoint（可选，但推荐）
rm -rf outputs/chengla-8B-gspo-warmup-fsdp/checkpoints/*

# 重新运行 Warmup 脚本
bash scripts/train/A3b_gspo/content_merge_trainning/A3b-warmup-gspo.sh
```

**预计时间**：约 12 小时

### 2. 监控关键指标

**前 50 步重点观察**：
```python
# 分数是否上升
critic/teacher_score_mean: -1.4 → 3.0 → 5.0 → 7.0
critic/student_score_mean: -2.5 → 1.0 → 3.0 → 4.0

# 正则化损失是否下降
critic/score_reg: 4.0 → 0.8 → 0.2 → 0.05

# 分数是否进入目标范围
critic/teacher_in_range_ratio: 0.0 → 0.5 → 0.8 → 0.95
critic/student_in_range_ratio: 0.0 → 0.6 → 0.9 → 0.98

# 相对关系是否保持
critic/score_diff: 1.0 → 2.0 → 3.0 → 3.5
critic/d_acc: 0.6 → 0.7 → 0.75 → 0.8
```

**异常情况处理**：

| 情况 | 症状 | 解决方案 |
|------|------|----------|
| 分数上升太慢 | Step 50 时 teacher_score < 3.0 | 增大 SCORE_REG_WEIGHT 到 0.2 |
| 分数震荡 | score_diff 波动 > 1.0 | 减小 SCORE_REG_WEIGHT 到 0.05 |
| ranking_loss 下降太慢 | Step 100 时 ranking_loss > 0.5 | 调整 temperature（见备选方案） |
| 分数超出范围 | teacher_score > 15 或 < 0 | 检查数据质量，可能需要调整范围 |

### 3. 验证效果（训练完成后）

**检查清单**：
- [ ] `critic/teacher_score_mean` 在 5-10 范围
- [ ] `critic/student_score_mean` 在 0-10 范围
- [ ] `critic/score_diff` > 0（Teacher > Student）
- [ ] `critic/d_acc` 在 0.5-0.7 范围（不要太高）
- [ ] `critic/teacher_in_range_ratio` > 0.9
- [ ] `critic/student_in_range_ratio` > 0.9
- [ ] `critic/score_reg` < 0.1（接近 0）

### 4. 调整超参数（如果需要）

**如果效果不理想**：

```python
# 调整 1：修改目标范围
TEACHER_SCORE_LOW = 6.0  # 从 5.0 提高到 6.0
TEACHER_SCORE_HIGH = 10.0  # 保持不变

# 调整 2：修改正则化权重
SCORE_REG_WEIGHT = 0.2  # 从 0.1 增大到 0.2（如果分数上升太慢）
SCORE_REG_WEIGHT = 0.05  # 从 0.1 减小到 0.05（如果分数震荡）

# 调整 3：修改 ranking_loss 权重
d_loss = 2.0 * ranking_loss + score_reg  # 从 3.0 减小到 2.0（如果 score_reg 影响太小）
d_loss = 5.0 * ranking_loss + score_reg  # 从 3.0 增大到 5.0（如果 score_reg 影响太大）
```

## 风险与缓解

### 风险 1：需要重新 Warmup

**风险**：当前 Critic 已经学到负分数，修改后需要重新训练

**缓解**：
- ✅ 这是必要的，当前分数确实不合理
- ✅ Warmup 相对较快（约 12 小时）
- ✅ 长期收益大于短期成本

### 风险 2：可能影响区分度

**风险**：约束分数范围可能降低 Critic 的区分能力

**缓解**：
- ✅ 使用软约束，不是硬截断
- ✅ ranking_loss 仍然是主要目标（权重 3.0）
- ✅ score_reg 只是辅助（权重 0.1）
- ✅ 10 分的范围足够区分不同质量

### 风险 3：超参数需要调整

**风险**：权重 0.1 可能不是最优的

**缓解**：
- ✅ 从 0.1 开始，观察前 50 步
- ✅ 如果分数上升太慢，增大到 0.2
- ✅ 如果分数震荡，减小到 0.05
- ✅ 监控 score_reg 的值，应该在 0.1-0.5 范围

### 风险 4：与现有训练冲突

**风险**：新的正则化可能与 ranking_loss 冲突

**缓解**：
- ✅ ranking_loss 权重更大（3.0 vs 0.1）
- ✅ 两者目标一致（都希望 teacher > student）
- ✅ score_reg 只是约束范围，不改变相对关系

## 技术细节

### 代码位置
- **文件**: `verl/verl/trainer/ppo/core_algos.py`
- **函数**: `compute_discriminator_loss`
- **行数**: 约 1500-1550 行

### 依赖关系
- **无新增依赖**：只使用了 PyTorch 内置函数
- **向后兼容**：保持了函数签名不变
- **接口稳定**：loss_info 字典只是新增字段，不影响现有代码

### 测试建议
```python
# 单元测试（可选）
def test_score_range_regularization():
    # 测试 1：分数在范围内，惩罚为 0
    teacher_score = torch.tensor([7.0, 8.0, 6.0])
    student_score = torch.tensor([3.0, 4.0, 5.0])
    # 预期：score_reg ≈ 0
    
    # 测试 2：分数低于范围，有惩罚
    teacher_score = torch.tensor([3.0, 4.0, 2.0])
    student_score = torch.tensor([-1.0, -2.0, 0.5])
    # 预期：score_reg > 0
    
    # 测试 3：分数高于范围，有惩罚
    teacher_score = torch.tensor([12.0, 15.0, 11.0])
    student_score = torch.tensor([11.0, 12.0, 13.0])
    # 预期：score_reg > 0
```

## 相关文档

- **设计文档**: `Critic分数范围引导设计方案.md`
- **影响分析**: `Critic负分数对Actor训练的影响分析.md`
- **d_acc 说明**: `d_acc在训练中的作用说明.md`
- **训练脚本**: `scripts/train/A3b_gspo/content_merge_trainning/A3b-warmup-gspo.sh`

## 总结

### 核心改进
1. ✅ 将 Score Drift Regularization 替换为 Score Range Regularization
2. ✅ 引导 Teacher 分数到 [5, 10] 范围
3. ✅ 引导 Student 分数到 [0, 10] 范围
4. ✅ 新增 6 个监控指标
5. ✅ 调整 adaptive temperature 映射范围（0.5*diff, [0.5, 3.0]）

### 预期收益
1. ✅ 分数有明确语义（5=及格，10=完美）
2. ✅ 解决负分数问题
3. ✅ 提高训练稳定性
4. ✅ 便于人工监控和调试

### 下一步
1. ⏳ 重新运行 Warmup 训练（约 12 小时）
2. ⏳ 监控前 50 步的指标变化
3. ⏳ 验证训练完成后的效果
4. ⏳ 根据需要调整超参数

---

**创建日期**: 2026-02-05  
**作者**: AI Assistant  
**状态**: 实施完成，待训练验证  
**最后更新**: 2026-02-05
