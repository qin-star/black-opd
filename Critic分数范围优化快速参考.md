# Critic 分数范围优化 - 快速参考

## ✅ 已完成的修改

### 核心改动
- **文件**: `verl/verl/trainer/ppo/core_algos.py`
- **函数**: `compute_discriminator_loss`
- **改动 1**: Score Drift Regularization → Score Range Regularization
- **改动 2**: Adaptive Temperature 映射调整（0.5*diff, [0.5, 3.0]）

### 目标范围
```python
Teacher: [5, 10]  # 高质量答案
Student: [0, 10]  # 质量不确定
```

### 分数语义
```
Teacher:
  10 = 完美  8 = 优秀  6 = 良好  5 = 及格

Student:
  10 = 完美  8 = 优秀  5 = 中等  2 = 较差  0 = 很差
```

## 🚀 下一步操作

### 1. 重新 Warmup（必须）
```bash
# 清理旧 checkpoint（推荐）
rm -rf outputs/chengla-8B-gspo-warmup-fsdp/checkpoints/*

# 重新训练
bash scripts/train/A3b_gspo/content_merge_trainning/A3b-warmup-gspo.sh
```

**预计时间**: 约 12 小时

### 2. 监控关键指标

#### 前 50 步重点观察
```python
# 分数是否上升到目标范围
critic/teacher_score_mean: -1.4 → 3.0 → 5.0 → 7.0 ✅
critic/student_score_mean: -2.5 → 1.0 → 3.0 → 4.0 ✅

# 正则化损失是否下降
critic/score_reg: 4.0 → 0.8 → 0.2 → 0.05 ✅

# 分数是否进入目标范围
critic/teacher_in_range_ratio: 0.0 → 0.5 → 0.8 → 0.95 ✅
critic/student_in_range_ratio: 0.0 → 0.6 → 0.9 → 0.98 ✅
```

#### 训练完成后验证
```python
# 检查清单
✅ teacher_score_mean 在 5-10 范围
✅ student_score_mean 在 0-10 范围
✅ score_diff > 0 (Teacher > Student)
✅ d_acc 在 0.5-0.7 范围
✅ teacher_in_range_ratio > 0.9
✅ student_in_range_ratio > 0.9
✅ score_reg < 0.1
```

## 🔧 异常情况处理

### 分数上升太慢
**症状**: Step 50 时 teacher_score < 3.0

**解决**: 增大正则化权重
```python
# 在 core_algos.py 中修改
SCORE_REG_WEIGHT = 0.2  # 从 0.1 增大到 0.2
```

### 分数震荡
**症状**: score_diff 波动 > 1.0

**解决**: 减小正则化权重
```python
SCORE_REG_WEIGHT = 0.05  # 从 0.1 减小到 0.05
```

### ranking_loss 下降太慢
**症状**: Step 100 时 ranking_loss > 0.5

**解决**: 调整 temperature
```python
# 方案 1: 调整自适应范围
adaptive_temp = torch.clamp(
    torch.tensor(0.5 * current_diff_abs, device=diff.device),
    min=0.5,  # 从 0.3 提高
    max=3.0   # 从 2.0 提高
)

# 方案 2: 使用固定值
temperature = 2.0
adaptive_temperature = False
```

## 📊 新增监控指标

```python
# 范围内比例
critic/teacher_in_range_ratio  # Teacher 在 [5,10] 的比例
critic/student_in_range_ratio  # Student 在 [0,10] 的比例

# 分数范围
critic/teacher_score_min  # Teacher 最小分数
critic/teacher_score_max  # Teacher 最大分数
critic/student_score_min  # Student 最小分数
critic/student_score_max  # Student 最大分数
```

## 📝 相关文档

- **详细说明**: `Critic指标优化完成说明.md`
- **设计方案**: `Critic分数范围引导设计方案.md`
- **影响分析**: `Critic负分数对Actor训练的影响分析.md`

## ⚠️ 重要提醒

1. **必须重新 Warmup**: 当前 Critic 已学到负分数，需要重新训练
2. **观察前 50 步**: 分数应该快速上升到目标范围
3. **不要跳过验证**: 训练完成后检查所有指标
4. **保存训练日志**: 便于后续分析和调试

---

**状态**: ✅ 代码修改完成，⏳ 等待训练验证  
**日期**: 2026-02-05
