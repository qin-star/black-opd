# Warmup 阶段指标分析与优化方案

## 日期
2026-01-28

## 当前训练指标分析 (Step 49)

### ✅ 已成功修复的问题

1. **EOS Token 问题完全解决**
   - 相同答案平均分差: **0.0000** ✅
   - 相同答案数量: 2/32
   - `_compute_last_token_mask` 方法正确跳过 EOS token
   - Student 和 Teacher 现在提取相同位置的 token

2. **梯度信号健康**
   - `grad_norm_before_clip`: **3.458** ✅
   - `grad_norm`: **3.458** (未触发裁剪)
   - 有效梯度远大于优化前的 < 0.1

### ⚠️ 核心问题：Critic 过度自信

#### 问题 1: d_acc 过高
```
当前值: 93.55%
目标值: 65-85%
问题: Critic 几乎能完美区分所有样本，说明区分能力过强
```

#### 问题 2: score_diff 过大
```
当前值: 9.81
目标值: 1-5
原始分差: 13.88
归一化分差: 0.291

Teacher 平均分: 5.51
Student 平均分: -4.30
分数差异: ~10 (过大)
Teacher > Student: 87.5% (过高)
```

#### 问题 3: Temperature 不足
```
当前 temperature: 2.0
sigmoid(9.81 / 2.0) = sigmoid(4.905) ≈ 0.993 (接近饱和)

需要更大的 temperature 来缓解梯度饱和
```

#### 问题 4: 分数范围未得到有效控制
```
Student 分数范围: [-14.0, 8.25]  (范围 ~22)
Teacher 分数范围: [-8.44, 19.0]  (范围 ~27)

虽然有 score_reg，但分数仍然分布过广
```

## 根本原因分析

### 1. 数据质量问题
- Teacher > Student 87.5% 说明数据中 teacher 质量明显更好
- 这导致 Critic 学到"所有 teacher 都好，所有 student 都差"的粗糙模式
- 缺乏细微的质量差异，Critic 无法学习精细区分

### 2. Temperature 设置不当
- 当前 temperature=2.0 对于 score_diff=9.81 来说太小
- 导致 sigmoid 函数饱和，梯度信号弱
- 需要根据实际 score_diff 动态调整

### 3. 正则化不足
- score_threshold=5.0 太高，很多极端值未被约束
- score_reg_weight=0.001 太小，约束力度不够
- 需要更强的正则化来控制分数范围

### 4. 缺少 Batch Normalization
- 当前直接使用原始分数计算 loss
- 不同 batch 间分数分布可能差异很大
- 缺少归一化导致训练不稳定

## 优化方案

### 方案 A: 增大 Temperature (立即实施) ⭐⭐⭐⭐⭐

**目标**: 缓解梯度饱和，让 Critic 学习更细微的差异

**修改位置**: `verl/verl/workers/critic/dp_critic.py` line 747

```python
# 当前代码
d_loss, loss_info = core_algos.compute_discriminator_loss(
    student_vpreds=student_vpreds,
    teacher_vpreds=teacher_vpreds,
    response_mask=response_mask,
    teacher_response_mask=teacher_response_mask,
    temperature=0.5,  # 基础 temperature
    adaptive_temperature=True,  # 启用自适应
)

# 优化后
d_loss, loss_info = core_algos.compute_discriminator_loss(
    student_vpreds=student_vpreds,
    teacher_vpreds=teacher_vpreds,
    response_mask=response_mask,
    teacher_response_mask=teacher_response_mask,
    temperature=5.0,  # 增大基础 temperature
    adaptive_temperature=False,  # 暂时关闭自适应，使用固定值
)
```

**理由**:
- 当前 score_diff=9.81，temperature=2.0 时 sigmoid(4.905)≈0.993 (饱和)
- 增大到 5.0 后 sigmoid(1.962)≈0.877 (不饱和)
- 梯度信号更强，Critic 可以学习更细微的差异

**预期效果**:
- ranking_loss 会增大（因为梯度更强）
- d_acc 会下降到 70-80%
- score_diff 会逐渐减小

### 方案 B: 增强 Score Regularization (立即实施) ⭐⭐⭐⭐⭐

**目标**: 更强地约束分数范围，防止极端值

**修改位置**: `verl/verl/trainer/ppo/core_algos.py` line 1490-1494

```python
# 当前代码
score_threshold = 5.0
teacher_extreme = torch.nn.functional.relu(teacher_score_raw.abs() - score_threshold)
student_extreme = torch.nn.functional.relu(student_score_raw.abs() - score_threshold)
score_reg = 0.001 * (teacher_extreme.pow(2).mean() + student_extreme.pow(2).mean())

# 优化后
score_threshold = 3.0  # 从 5.0 降低到 3.0
teacher_extreme = torch.nn.functional.relu(teacher_score_raw.abs() - score_threshold)
student_extreme = torch.nn.functional.relu(student_score_raw.abs() - score_threshold)
score_reg = 0.01 * (teacher_extreme.pow(2).mean() + student_extreme.pow(2).mean())  # 从 0.001 增加到 0.01
```

**理由**:
- 当前分数范围 [-14, 19]，threshold=5.0 只约束了极端值
- 降低 threshold 到 3.0，更多分数会被约束
- 增大权重到 0.01，约束力度增强 10 倍

**预期效果**:
- 分数范围会缩小到 [-10, 10]
- score_reg 会增大（因为更多值被约束）
- 训练更稳定

### 方案 C: 添加 Batch Normalization (立即实施) ⭐⭐⭐⭐

**目标**: 自动适应不同的分数范围，稳定训练

**修改位置**: `verl/verl/trainer/ppo/core_algos.py` line 1450-1470

```python
# 在计算 diff 之前添加 batch normalization
def compute_discriminator_loss(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    margin: float = 0.5,
    temperature: float = 0.5,
    adaptive_temperature: bool = False,
    use_batch_norm: bool = True,  # 新增参数
) -> tuple:
    # ... 前面的代码保持不变 ...
    
    # 混合策略计算分数
    raw_weight = 0.7
    norm_weight = 0.3
    teacher_score = raw_weight * teacher_score_raw + norm_weight * teacher_score_norm
    student_score = raw_weight * student_score_raw + norm_weight * student_score_norm
    
    # 新增：Batch Normalization
    if use_batch_norm:
        # 合并所有分数进行归一化
        all_scores = torch.cat([teacher_score, student_score])
        mean = all_scores.mean()
        std = all_scores.std() + 1e-8
        
        # 归一化
        teacher_score_normalized = (teacher_score - mean) / std
        student_score_normalized = (student_score - mean) / std
        
        # 使用归一化后的分数计算 diff
        diff = teacher_score_normalized - student_score_normalized
        
        # 记录归一化信息（用于监控）
        batch_norm_info = {
            "batch_mean": mean.detach().item(),
            "batch_std": std.detach().item(),
        }
    else:
        diff = teacher_score - student_score
        batch_norm_info = {}
    
    # ... 后面的代码保持不变 ...
    
    loss_info.update(batch_norm_info)
    return d_loss, loss_info
```

**调用位置**: `verl/verl/workers/critic/dp_critic.py` line 747

```python
d_loss, loss_info = core_algos.compute_discriminator_loss(
    student_vpreds=student_vpreds,
    teacher_vpreds=teacher_vpreds,
    response_mask=response_mask,
    teacher_response_mask=teacher_response_mask,
    temperature=5.0,
    adaptive_temperature=False,
    use_batch_norm=True,  # 启用 batch normalization
)
```

**理由**:
- 自动适应不同 batch 的分数分布
- 保持相对顺序，不改变判别逻辑
- 稳定训练，减小方差

**预期效果**:
- diff 会被归一化到合理范围（通常 -3 到 3）
- 不同 batch 间的训练更一致
- d_acc 会更稳定

### 方案 D: 调整自适应 Temperature 策略 (可选) ⭐⭐⭐

**目标**: 根据实际 score_diff 动态调整 temperature

**修改位置**: `verl/verl/trainer/ppo/core_algos.py` line 1475-1490

```python
# 当前代码
if adaptive_temperature:
    current_diff_abs = diff.abs().mean().item()
    adaptive_temp = torch.clamp(
        torch.tensor(0.3 + current_diff_abs * 0.7, device=diff.device),
        min=0.3,
        max=2.0
    )
    temperature = adaptive_temp.item()

# 优化后
if adaptive_temperature:
    current_diff_abs = diff.abs().mean().item()
    # 新的映射函数：
    # - diff=0.5  → T=1.0  (标准梯度)
    # - diff=1.0  → T=2.0  (中等梯度)
    # - diff=5.0  → T=5.0  (弱梯度，精细区分)
    # - diff=10.0 → T=7.0  (很弱梯度，防止过拟合)
    adaptive_temp = torch.clamp(
        torch.tensor(1.0 + current_diff_abs * 0.6, device=diff.device),
        min=1.0,
        max=7.0
    )
    temperature = adaptive_temp.item()
```

**理由**:
- 当前映射范围 [0.3, 2.0] 对于大 diff 不够
- 新映射范围 [1.0, 7.0] 更适合当前情况
- 随着训练进行，diff 减小，temperature 自动降低

**预期效果**:
- 训练初期 temperature 大（5-7），梯度弱，防止过拟合
- 训练后期 temperature 小（1-2），梯度强，精细学习
- 自动适应训练进度

## 实施计划

### 阶段 1: 立即实施 (优先级最高)

1. **增大 Temperature 到 5.0** (方案 A)
   - 修改 `dp_critic.py` line 747
   - 关闭自适应，使用固定值 5.0

2. **增强 Score Regularization** (方案 B)
   - 修改 `core_algos.py` line 1490-1494
   - threshold: 5.0 → 3.0
   - weight: 0.001 → 0.01

3. **添加 Batch Normalization** (方案 C)
   - 修改 `core_algos.py` 添加 use_batch_norm 参数
   - 修改 `dp_critic.py` 启用 batch normalization

### 阶段 2: 观察效果 (100-200 步)

**关键指标监控**:
```
d_acc: 应该从 93.55% 降到 70-80%
score_diff: 应该从 9.81 降到 3-5
ranking_loss: 可能会增大（正常现象）
score_reg: 会增大（因为约束更强）
grad_norm: 应该保持在 1-5 范围
```

**健康训练的特征**:
- d_acc 在 70-80% 范围
- score_diff 逐渐减小
- ranking_loss 稳定或缓慢增大
- 没有梯度爆炸或消失

### 阶段 3: 进一步优化 (可选)

如果效果不理想，考虑：

1. **进一步增大 Temperature**
   - 如果 d_acc 仍然 > 85%，增大到 7.0

2. **启用自适应 Temperature** (方案 D)
   - 如果固定 temperature 效果好，可以尝试自适应

3. **调整数据质量**
   - 运行 `tools/analyze_data_quality.py` 分析数据
   - 如果 teacher-student 差异确实很大，考虑调整数据

## 预期效果对比

### 修复前 (Step 49)
```
d_acc: 93.55%
score_diff: 9.81
teacher_score: 5.51
student_score: -4.30
分数范围: [-14, 19]
temperature: 2.0
```

### 修复后 (预期 Step 150)
```
d_acc: 70-80%
score_diff: 3-5
teacher_score: 2-3
student_score: -1 to 0
分数范围: [-10, 10]
temperature: 5.0
```

### 长期效果 (预期 Step 500+)
```
d_acc: 70-75%
score_diff: 1-3
teacher_score: 1-2
student_score: 0-1
分数范围: [-5, 5]
format_reward: 持续提升
```

## 验证方法

### 1. 快速验证 (每 10 步)
```bash
# 观察训练日志
tail -f logs/training.log | grep "critic/"
```

**检查点**:
- [ ] d_acc 开始下降
- [ ] score_diff 开始减小
- [ ] ranking_loss 变化合理
- [ ] 没有梯度爆炸

### 2. TensorBoard 监控
```bash
tensorboard --logdir=tensorboard/ --port=6022
```

**关键曲线**:
- `critic/d_acc`: 应该从 93% 降到 70-80%
- `critic/score_diff`: 应该从 9.8 降到 3-5
- `critic/ranking_loss`: 可能增大（正常）
- `critic/grad_norm`: 应该稳定在 1-5

### 3. 详细日志分析
```bash
# 查看打分详情
tail -f logs/critic_scoring_details/*.log
```

**检查点**:
- [ ] 相同答案分差仍然 < 0.5
- [ ] Teacher > Student 比例降到 70-80%
- [ ] 分数范围缩小

## 故障排查

### 问题 1: d_acc 仍然很高 (> 90%)

**可能原因**:
- Temperature 仍然不够大
- 数据质量差异确实很大

**解决方案**:
1. 进一步增大 temperature 到 7.0
2. 运行数据质量分析工具
3. 考虑使用更强的 batch normalization

### 问题 2: score_diff 不变化

**可能原因**:
- 梯度仍然过小
- Batch normalization 未生效

**解决方案**:
1. 检查 `use_batch_norm=True` 是否生效
2. 检查 grad_norm，如果 < 0.5，提高学习率
3. 检查 loss 是否正常反向传播

### 问题 3: 梯度爆炸

**症状**: grad_norm > 10.0

**解决方案**:
1. 降低学习率到 5e-6
2. 保持 grad_clip=1.0
3. 检查数据是否有异常值

### 问题 4: d_acc 过低 (< 60%)

**可能原因**:
- Temperature 过大
- Batch normalization 过度归一化

**解决方案**:
1. 降低 temperature 到 3.0
2. 检查数据质量
3. 考虑关闭 batch normalization

## 总结

### 核心问题
- Critic 过度自信 (d_acc=93.55%)
- 分数差异过大 (score_diff=9.81)
- Temperature 不足 (2.0)
- 正则化不足

### 解决方案
1. ✅ 增大 Temperature 到 5.0
2. ✅ 增强 Score Regularization (threshold: 5.0→3.0, weight: 0.001→0.01)
3. ✅ 添加 Batch Normalization
4. ⚠️ 可选：调整自适应 Temperature 策略

### 预期收益
- d_acc 从 93.55% 降到 70-80%
- score_diff 从 9.81 降到 3-5
- 训练更稳定，Critic 学习更细微的差异
- Student 质量持续提升

### 下一步
1. 实施方案 A、B、C
2. 观察 100-200 步的效果
3. 根据效果决定是否需要方案 D
4. 持续监控训练指标

---

**创建日期**: 2026-01-28
**最后更新**: 2026-01-28
**状态**: 待实施
