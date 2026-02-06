# Warmup 阶段优化快速参考

## 📊 当前问题 (Step 49)

```
❌ d_acc: 93.55% (目标: 70-80%)
❌ score_diff: 9.81 (目标: 3-5)
❌ Teacher > Student: 87.5% (目标: 70-80%)
✅ EOS Token 问题: 已解决 (相同答案分差 0.0)
✅ 梯度信号: 健康 (grad_norm 3.458)
```

## 🔧 已实施的优化

### 1. Temperature: 0.5 → 5.0 ⭐⭐⭐⭐⭐
```python
# verl/verl/workers/critic/dp_critic.py line 747
temperature=5.0,  # 从 0.5 增大到 5.0
adaptive_temperature=False,  # 关闭自适应
```

**理由**: sigmoid(9.81/2.0)≈0.993 饱和 → sigmoid(9.81/5.0)≈0.877 不饱和

### 2. Score Regularization 增强 ⭐⭐⭐⭐⭐
```python
# verl/verl/trainer/ppo/core_algos.py line 1490-1494
score_threshold = 3.0  # 从 5.0 降低
score_reg = 0.01 * (...)  # 从 0.001 增大 10 倍
```

**理由**: 约束更多极端值，控制分数范围在 [-10, 10]

### 3. Batch Normalization 启用 ⭐⭐⭐⭐
```python
# verl/verl/workers/critic/dp_critic.py line 747
use_batch_norm=True,  # 启用归一化
```

**理由**: 自动适应不同 batch 的分数分布，稳定训练

## 📈 预期效果

### 短期 (100-200 步)
```
d_acc: 93.55% → 70-80%
score_diff: 9.81 → 5-7
ranking_loss: 可能增大 (正常)
score_reg: 会增大 (因为约束更强)
```

### 中期 (500 步)
```
d_acc: 稳定在 70-75%
score_diff: 3-5
分数范围: [-10, 10]
```

### 长期 (1000+ 步)
```
d_acc: 70-75%
score_diff: 1-3
分数范围: [-5, 5]
format_reward: 持续提升
```

## 🔍 监控指标

### 核心指标 (每 10 步检查)
```bash
# 观察训练日志
tail -f logs/training.log | grep "critic/"
```

**健康训练的特征**:
- ✅ d_acc 在 70-80% 范围
- ✅ score_diff 逐渐减小
- ✅ ranking_loss 稳定或缓慢增大
- ✅ grad_norm 在 1-5 范围
- ✅ 没有梯度爆炸或消失

### TensorBoard 监控
```bash
tensorboard --logdir=tensorboard/ --port=6022
```

**关键曲线**:
- `critic/d_acc`: 应该下降
- `critic/score_diff`: 应该减小
- `critic/ranking_loss`: 可能增大
- `critic/grad_norm`: 应该稳定

## ⚠️ 故障排查

### 问题 1: d_acc 仍然 > 90%
**解决**: 进一步增大 temperature 到 7.0

### 问题 2: score_diff 不变化
**解决**: 检查 grad_norm，如果 < 0.5，提高学习率

### 问题 3: 梯度爆炸 (grad_norm > 10)
**解决**: 降低学习率到 5e-6

### 问题 4: d_acc < 60%
**解决**: 降低 temperature 到 3.0

## 📝 验证清单

运行验证脚本：
```bash
python tools/verify_warmup_optimization.py
```

应该看到：
- ✅ Temperature 设置: 通过
- ✅ Score Regularization 设置: 通过
- ✅ Batch Normalization 实现: 通过
- ✅ 文档完整性: 通过

## 🚀 下一步

1. **重新启动训练**
   ```bash
   bash scripts/train/A3b_gspo/content_merge_trainning/A3b-warmup-gspo-optimized.sh
   ```

2. **观察前 100 步**
   - d_acc 是否开始下降
   - score_diff 是否开始减小
   - 没有梯度爆炸

3. **持续监控**
   - 每 50 步检查一次指标
   - 记录关键变化
   - 必要时调整参数

## 📚 相关文档

- `Warmup阶段指标分析与优化方案.md` - 完整分析和方案
- `Critic训练优化总结.md` - 之前的优化历史
- `EOS_Token问题修复完成说明.md` - EOS token 修复
- `Critic分数分布分析与优化.md` - 分数分布分析

---

**创建日期**: 2026-01-28
**状态**: ✅ 已实施，待验证
