# GAD 训练指标分析指南

## 文档目的

这是一份**标准化的训练指标分析指南**，用于：
1. 快速诊断 GAD 训练是否健康
2. 避免单一指标误导导致的错误决策
3. 提供基于证据的调优建议

## 快速开始

### 1. 运行分析脚本

```bash
# 进入项目目录
cd E:\LLM-trainning\gad_gspo_b300

# 运行分析脚本
python analyze_tensorboard.py

# 或者指定日志文件
python analyze_tensorboard.py --log-file logs/events.out.tfevents.xxx
```

### 2. 查看关键指标

脚本会自动输出以下关键指标：
- critic/d_acc
- critic/score_diff
- critic/ranking_loss
- critic/teacher_value_mean
- critic/student_value_mean
- critic/score_reg
- critic/diff_penalty
- format/reward_avg

### 3. 使用本指南进行分析

根据输出的指标值，按照本指南的决策树进行判断。

---

## 核心分析原则

### ⚠️ 重要：不要只看单一指标！

**错误做法**：
```
❌ d_acc = 97% → 立即修改 temperature
```

**正确做法**：
```
✅ d_acc = 97% → 检查 score_diff → 检查 ranking_loss → 综合判断
```

### 指标优先级

```
1. score_diff (最重要)          ← 判断质量差距
2. ranking_loss (最重要)        ← 判断收敛状态
3. 绝对分数范围 (重要)          ← 判断数值漂移
4. format/reward_avg (重要)     ← 判断 student 进步
5. d_acc (参考)                 ← 仅作参考，不作为主要依据
```

---

## 标准分析流程

### Step 1: 提取关键指标

运行分析脚本后，记录以下数值：

```
critic/d_acc:              _____
critic/score_diff:         _____
critic/ranking_loss:       _____
critic/teacher_value_mean: _____
critic/student_value_mean: _____
critic/score_reg:          _____
critic/diff_penalty:       _____
format/reward_avg:         _____
```

### Step 2: 计算健康度评分

#### 2.1 score_diff 评分（40分）

| score_diff 范围 | 评分 | 状态 |
|----------------|------|------|
| < 0.1 | 40分 | 优秀 - Student 已接近 Teacher |
| 0.1 - 0.5 | 30分 | 良好 - Student 质量不错 |
| 0.5 - 1.0 | 20分 | 可接受 - Student 还在学习 |
| 1.0 - 2.0 | 10分 | 需要关注 - 差距较大 |
| > 2.0 | 0分 | 有问题 - 差距过大 |

**你的得分**：_____

#### 2.2 ranking_loss 评分（30分）

| ranking_loss 范围 | 评分 | 状态 |
|------------------|------|------|
| 0.65 - 0.72 | 30分 | 优秀 - 接近理论最优 log(2)≈0.693 |
| 0.50 - 0.65 或 0.72 - 0.80 | 20分 | 良好 - 在合理范围 |
| 0.30 - 0.50 | 10分 | 需要关注 - 区分太容易 |
| < 0.30 或 > 0.80 | 0分 | 有问题 - 异常 |

**你的得分**：_____

#### 2.3 绝对分数评分（20分）

| 条件 | 评分 | 状态 |
|------|------|------|
| 两个分数都在 [-2, 2] | 20分 | 优秀 - 无数值漂移 |
| 两个分数都在 [-5, 5] | 10分 | 可接受 - 轻微漂移 |
| 任一分数 > 5 或 < -5 | 0分 | 有问题 - 数值漂移 |

**你的得分**：_____

#### 2.4 d_acc 评分（10分）

| d_acc 范围 | 评分 | 说明 |
|-----------|------|------|
| < 0.70 | 10分 | 理想状态 |
| 0.70 - 0.85 | 8分 | 良好状态 |
| 0.85 - 0.95 | 5分 | 可接受 |
| > 0.95 | 0分 | 需要结合其他指标判断 |

**你的得分**：_____

#### 总分计算

```
总分 = score_diff 评分 + ranking_loss 评分 + 绝对分数评分 + d_acc 评分
     = _____ + _____ + _____ + _____
     = _____ / 100
```

### Step 3: 健康度判断

| 总分范围 | 健康度 | 建议 |
|---------|--------|------|
| 80-100 | ✅ 优秀 | 继续当前训练，不需要修改 |
| 60-79 | ✅ 良好 | 训练基本健康，可以继续 |
| 40-59 | ⚠️ 需要关注 | 观察更多步数，可能需要调整 |
| 20-39 | ⚠️ 需要调整 | 存在明显问题，建议调整参数 |
| < 20 | ❌ 需要修复 | 训练有严重问题，必须修改 |

**你的健康度**：_____

---

## 典型场景分析

### 场景1：d_acc 高但训练健康 ✅

#### 指标特征
```
critic/d_acc = 0.97              ⚠️ 高
critic/score_diff = 0.08         ✅ 很小
critic/ranking_loss = 0.68       ✅ 接近 log(2)
teacher_value_mean = 0.43        ✅ 合理范围
student_value_mean = -0.34       ✅ 合理范围
```

#### 诊断结论
**这是正常的！不需要修改代码。**

#### 原因分析
1. **score_diff 很小**（0.08）说明 Teacher 和 Student 在归一化质量上几乎相同
2. **ranking_loss ≈ log(2)** 说明 Critic 已经接近无法区分两者
3. **d_acc 高**是因为累积效应：
   ```
   每个 token 差异 = 0.08
   序列长度 = 10
   累积差异 = 0.08 × 10 = 0.8
   → Teacher 总是略高于 Student
   → d_acc = 97%
   ```

#### 建议
- ✅ 继续当前训练
- ✅ 不要修改任何参数
- ✅ 观察 score_diff 是否继续缩小

---

### 场景2：d_acc 高且 score_diff 大 ❌

#### 指标特征
```
critic/d_acc = 0.97              ⚠️ 高
critic/score_diff = 2.5          ❌ 很大
critic/ranking_loss = 0.25       ❌ 很低
teacher_value_mean = 0.50        ✅ 合理
student_value_mean = -2.00       ❌ 很低
```

#### 诊断结论
**有问题！需要调整。**

#### 原因分析
1. **score_diff 很大**（2.5）说明 Teacher 远好于 Student
2. **ranking_loss 很低**（0.25）说明区分太容易
3. 可能原因：
   - Student 质量确实很差
   - Format reward 过于严格
   - Temperature 太小

#### 建议
1. **优先检查 format/reward_avg**：
   - 如果 < -0.5 → 降低 format reward 惩罚
   - 如果正常 → 继续观察

2. **如果确认需要调整**：
   - 增大 temperature：2.0 → 3.0
   - 观察 100-200 steps
   - 根据效果继续调整

---

### 场景3：数值漂移 ❌

#### 指标特征
```
critic/d_acc = 0.97              ⚠️ 高
critic/score_diff = 1.5          ⚠️ 较大
teacher_value_mean = 8.5         ❌ 过大
student_value_mean = -7.0        ❌ 过小
critic/score_reg = 0.001         ❌ 太小
critic/diff_penalty = 0.8        ❌ 频繁触发
```

#### 诊断结论
**Loss 设计有问题！需要修改。**

#### 原因分析
1. **绝对分数过大**（> 5）说明发生数值漂移
2. **score_reg 太小**（0.001）无法约束分数
3. **diff_penalty 频繁触发**（0.8）但无效

#### 建议
1. **增大 score_reg 权重**：
   ```python
   # 在 core_algos.py 中
   score_reg = 0.02 * (...)  # 从 0.005 增大到 0.02
   ```

2. **降低 diff_penalty 阈值**：
   ```python
   # 在 core_algos.py 中
   diff_penalty = relu(diff - 0.5)  # 从 1.5 降低到 0.5
   ```

3. **可能需要增大 temperature**：
   ```python
   temperature = 3.0  # 从 2.0 增大到 3.0
   ```

---

### 场景4：ranking_loss 异常 ⚠️

#### 指标特征
```
critic/d_acc = 0.97              ⚠️ 高
critic/score_diff = 0.5          ⚠️ 中等
critic/ranking_loss = 0.15       ❌ 异常低
```

#### 诊断结论
**Temperature 太小，需要调整。**

#### 原因分析
1. **ranking_loss 太低**（0.15 << 0.693）说明 loss 对差异过于敏感
2. Critic 倾向于放大 Teacher 和 Student 的差距

#### 建议
**增大 temperature**：
```python
# 在 core_algos.py 中
temperature = 5.0  # 从 2.0 增大到 5.0
```

**预期效果**：
- ranking_loss 应该增大到 0.5-0.7
- d_acc 应该下降到 70-85%

---

## 关键指标解释

### 1. score_diff（最重要）

**定义**：Teacher 和 Student 的归一化分数差异（平均每个 token）

**计算公式**：
```python
teacher_score = teacher_score_raw / teacher_mask_sum  # 归一化
student_score = student_score_raw / student_mask_sum
score_diff = (teacher_score - student_score).mean()
```

**理想值**：< 0.1（说明质量差距很小）

**解释**：
- score_diff = 0.08 → 每个 token 平均差 0.008
- 这是非常小的差异
- 说明 Student 已经接近 Teacher

### 2. ranking_loss（最重要）

**定义**：Discriminator 的排序损失

**计算公式**：
```python
scaled_diff = (teacher_score - student_score) / temperature
ranking_loss = -log(sigmoid(scaled_diff)).mean()
```

**理论最优值**：log(2) ≈ 0.693

**解释**：
- 当 Teacher 和 Student 质量相当时：
  ```
  P(teacher > student) ≈ 0.5
  ranking_loss = -log(0.5) = log(2) ≈ 0.693
  ```
- ranking_loss ≈ 0.68 说明 Critic 接近无法区分
- 这是收敛的标志！

### 3. raw_score_diff vs score_diff

**关系**：
```
raw_score_diff ≈ score_diff × avg_sequence_length
```

**示例**：
```
score_diff = 0.08 (归一化差异)
avg_length = 10
raw_score_diff = 0.08 × 10 = 0.8 (累积差异)
```

**为什么 d_acc 基于 raw_score_diff？**
- d_acc 判断的是序列级别的排序
- 即使每个 token 差异很小（0.08）
- 累积后差异被放大（0.8）
- 所以 Teacher 总是略高于 Student
- d_acc = 97% 是正常的

### 4. 绝对分数范围

**理想范围**：
- teacher_value_mean: -2 到 2
- student_value_mean: -2 到 2

**如果超出范围**：
- > 5 或 < -5 → 数值漂移
- 需要增大 score_reg 权重

### 5. Loss 组件

**理想比例**：
```
ranking_loss: 主导（> 90%）
score_reg: 正则化（< 5%）
diff_penalty: 惩罚（< 5%）
```

**如果比例失衡**：
- ranking_loss < 80% → 正则化过强
- diff_penalty > 10% → 频繁触发，需要调整

---

## 参数调整指南

### 何时调整 temperature？

#### 调整条件
```
ranking_loss < 0.3  且  d_acc > 0.95
```

#### 调整方法
```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line ~1467

# 当前值
temperature = 2.0

# 调整建议
if ranking_loss < 0.3:
    temperature = 5.0  # 增大到 5.0
elif ranking_loss < 0.5:
    temperature = 3.0  # 增大到 3.0
```

#### 预期效果
- ranking_loss 增大到 0.5-0.7
- d_acc 下降到 70-85%
- score_diff 可能略微增大（正常）

#### 观察周期
- 训练 100-200 steps
- 观察指标变化
- 根据效果继续调整

### 何时调整 score_reg？

#### 调整条件
```
|teacher_value_mean| > 5  或  |student_value_mean| > 5
```

#### 调整方法
```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line ~1472

# 当前值
score_reg = 0.005 * (...)

# 调整建议
if 绝对分数 > 5:
    score_reg = 0.02 * (...)  # 增大到 0.02
elif 绝对分数 > 3:
    score_reg = 0.01 * (...)  # 增大到 0.01
```

#### 预期效果
- 绝对分数回到 -2 到 2 范围
- 数值漂移得到控制

### 何时调整 diff_penalty？

#### 调整条件
```
diff_penalty > 0.3  (频繁触发)
```

#### 调整方法
```python
# 在 verl/verl/trainer/ppo/core_algos.py 中
# Line ~1475

# 当前值
diff_penalty = relu(diff - 1.5).pow(2).mean()

# 调整建议
if diff_penalty > 0.5:
    diff_penalty = relu(diff - 0.5).pow(2).mean()  # 降低阈值到 0.5
elif diff_penalty > 0.3:
    diff_penalty = relu(diff - 1.0).pow(2).mean()  # 降低阈值到 1.0
```

#### 预期效果
- diff_penalty 减小到 < 0.3
- Critic 不会过度放大差距

### 何时调整 format reward？

#### 调整条件
```
format/reward_avg < -0.5  且  score_diff > 1.0
```

#### 调整方法
```python
# 在 verl/verl/utils/reward_score/gad_format_reward.py 中

# 减小惩罚系数
def compute_format_score(solution_str, ground_truth=""):
    # ... 检测逻辑 ...
    
    # 减小惩罚力度
    if json_issue:
        score -= json_issue["penalty"] * 0.5  # 乘以 0.5
    
    if lang_issue:
        score -= lang_issue["penalty"] * 0.5
    
    if content_issue:
        score -= content_issue["penalty"] * 0.5
```

#### 预期效果
- format/reward_avg 提升
- Student 质量改善
- score_diff 缩小

---

## 工具使用指南

### 1. 分析脚本：analyze_tensorboard.py

#### 功能
- 自动提取关键指标
- 计算统计量（均值、标准差、范围）
- 输出诊断建议

#### 使用方法
```bash
# 基本用法
python analyze_tensorboard.py

# 指定日志文件
python analyze_tensorboard.py --log-file logs/events.out.tfevents.xxx

# 输出到文件
python analyze_tensorboard.py > analysis_report.txt
```

#### 输出示例
```
critic/d_acc:
  最新值: 0.9824
  平均值: 0.9701 ± 0.0148
  范围: [0.9404, 0.9961]
  数据点数: 306

critic/score_diff:
  最新值: 0.0730
  平均值: 0.0799 ± 0.0290
  范围: [0.0299, 0.1573]
  数据点数: 306
```

### 2. 诊断工具：diagnose_high_dacc.py

#### 功能
- 专门诊断 d_acc 高的问题
- 提供详细的修复建议
- 对比修复前后的效果

#### 使用方法
```bash
# 单次诊断
python tools/diagnose_high_dacc.py --log-dir logs/your_run

# 对比修复前后
python tools/diagnose_high_dacc.py \
    --before-log-dir logs/before_fix \
    --after-log-dir logs/after_fix
```

### 3. TensorBoard 可视化

#### 启动 TensorBoard
```bash
tensorboard --logdir=logs --port=6006
```

#### 重点关注的图表
1. **critic/score_diff**：观察趋势是否下降
2. **critic/ranking_loss**：观察是否接近 0.693
3. **format/reward_avg**：观察 student 质量是否提升
4. **critic/d_acc**：仅作参考

#### 如何判断趋势
- **下降趋势**：✅ 好（score_diff, policy_loss）
- **上升趋势**：✅ 好（format_reward, ranking_loss）
- **稳定趋势**：✅ 好（绝对分数）
- **震荡趋势**：⚠️ 需要关注（可能不稳定）

---

## 常见问题 FAQ

### Q1: d_acc 一直很高（>95%），是否有问题？

**A**: 不一定。需要检查 score_diff：
- 如果 score_diff < 0.1 → ✅ 正常，不需要修改
- 如果 score_diff > 1.0 → ⚠️ 可能有问题，需要调整

### Q2: ranking_loss 一直在 0.68 左右，是否收敛了？

**A**: 是的！ranking_loss ≈ log(2) ≈ 0.693 是理论最优值，说明：
- Critic 已经接近无法区分 Teacher 和 Student
- 这是收敛的标志
- 可以考虑结束训练或调整其他参数

### Q3: score_diff 很小但 d_acc 还是很高，为什么？

**A**: 这是累积效应：
```
score_diff = 0.08 (每个 token 的平均差异)
序列长度 = 10
累积差异 = 0.08 × 10 = 0.8

虽然每个 token 差异很小，但累积后：
Teacher 总分 = 5.0
Student 总分 = 4.2
→ Teacher 总是略高
→ d_acc = 97%
```

这是正常的！

### Q4: 什么时候应该停止训练？

**A**: 满足以下条件之一：
1. **score_diff < 0.05** 且 **ranking_loss ≈ 0.69**
   - Student 已经非常接近 Teacher
   - 继续训练收益递减

2. **format/reward_avg 不再提升**
   - Student 质量已经达到瓶颈
   - 需要调整其他策略

3. **训练步数达到预设目标**
   - 例如 1000 steps 或 5000 steps

### Q5: 如何判断是 Critic 的问题还是 Actor 的问题？

**A**: 检查以下指标：

**如果是 Critic 的问题**：
```
✓ format/reward_avg 在提升
✓ actor/policy_loss 在下降
✗ score_diff 不收敛
✗ ranking_loss 异常
→ 调整 Critic 参数（temperature, score_reg）
```

**如果是 Actor 的问题**：
```
✗ format/reward_avg 不提升
✗ actor/policy_loss 不下降
✓ score_diff 正常
✓ ranking_loss 正常
→ 检查 Actor 配置（学习率、KL 系数）
```

### Q6: 修改参数后多久能看到效果？

**A**: 
- **Temperature 调整**：100-200 steps
- **Score_reg 调整**：50-100 steps
- **Format reward 调整**：200-500 steps

建议：
- 每次只调整一个参数
- 观察足够的步数
- 根据效果继续调整

---

## 检查清单

### 训练开始前

- [ ] 确认 TensorBoard 日志路径正确
- [ ] 确认分析脚本可以正常运行
- [ ] 记录初始参数配置（temperature, score_reg 等）

### 训练过程中（每 100 steps）

- [ ] 运行 `analyze_tensorboard.py` 提取指标
- [ ] 记录 score_diff 和 ranking_loss
- [ ] 检查 format/reward_avg 是否提升
- [ ] 观察 TensorBoard 曲线趋势

### 发现异常时

- [ ] 不要立即修改代码
- [ ] 收集所有关键指标
- [ ] 使用本指南的决策树判断
- [ ] 如果需要修改，先小幅调整
- [ ] 观察 100-200 steps 后再决定

### 训练结束后

- [ ] 保存最终的指标数据
- [ ] 记录最终的参数配置
- [ ] 评估 Student 的实际质量
- [ ] 总结经验教训

---

## 参考资料

### 相关文档
- `训练指标综合分析报告.md` - 实际案例分析
- `多指标综合分析框架.md` - 理论框架
- `d_acc持续高位问题深度诊断.md` - 问题诊断
- `训练健康度检查清单.md` - 评分系统

### 相关代码
- `verl/verl/trainer/ppo/core_algos.py` - Loss 计算
- `verl/verl/workers/critic/dp_critic.py` - Critic 训练
- `verl/verl/utils/reward_score/gad_format_reward.py` - Format reward

### 分析工具
- `analyze_tensorboard.py` - 主分析脚本
- `tools/diagnose_high_dacc.py` - d_acc 诊断工具
- `tools/verify_critic_order_fix.py` - 对比工具

---

## 版本历史

- **v1.0** (2025-01-22)
  - 初始版本
  - 基于实际训练数据总结
  - 包含完整的分析流程和决策树

---

## 贡献者

如果你在使用过程中发现问题或有改进建议，请：
1. 记录具体的训练数据和指标
2. 说明遇到的问题和期望的行为
3. 提供改进建议

这将帮助我们不断完善这份指南。
