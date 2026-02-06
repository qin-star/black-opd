# Critic 评分机制修改说明 - 平均值方案

## 修改日期
2026-01-28

## 问题回顾

### 发现的问题

通过训练日志分析，发现 Critic 使用 Last Token 机制存在严重缺陷：

**案例 1: 标点符号差异导致巨大分差**
```
Student: 男性，...，体重...  (逗号)  → Score: -0.56
Teacher: 男性，...。体重...  (句号)  → Score:  2.50
差异: 仅一个标点符号 → 分差 3.06 (不合理)
```

**案例 2: 完整回答反而得负分**
```
Student: 根据文章，因为各种原因无法回家团聚的人的年夜饭方式正在逐渐改变，包括网络年夜饭和朋友年夜饭。
  - 内容完整、准确、语义正确
  - Score: -2.33 ❌

Teacher: 网络年夜饭、朋友年夜饭。
  - 内容简洁
  - Score: 2.84 ✅

差异: Student 回答更完整 → 分数反而更低 (完全不合理)
```

### 根本原因

**Last Token 机制的致命缺陷**:

1. **只看最后一个 token 的 value**
   - 完全忽略前面所有 token 的内容
   - 依赖未训练模型中随机初始化的 value

2. **梯度只反向传播到最后一个 token**
   ```python
   score = values[last_token]
   loss.backward()
   # 只有 values[last_token] 有梯度
   # values[0:-1] 没有梯度，参数不更新
   ```

3. **Critic 无法学习语义**
   - 模型不需要"阅读"内容
   - 只是在比较随机的 token value
   - 学到的是位置/标点符号的模式，而非内容质量

## 修改方案

### 核心思路

**从 Last Token 改为平均值**，强制模型通过梯度反向传播学习整个序列的语义。

### 修改内容

**文件**: `verl/verl/workers/critic/dp_critic.py`

**修改位置**: `_forward_micro_batch` 方法

#### 修改前 (Last Token)

```python
# 旧代码
response_mask = attention_mask[:, -response_length:]
if compute_teacher:
    responses_for_mask = micro_batch["teacher_response"]
else:
    responses_for_mask = micro_batch["responses"]

# 计算 last token mask（跳过 EOS）
last_token_mask = self._compute_last_token_mask(
    responses_for_mask, response_mask, compute_teacher
)

# 只保留最后一个 token 的 value
values = values * last_token_mask.type_as(values)
```

#### 修改后 (平均值)

```python
# 新代码
response_mask = attention_mask[:, -response_length:]

# 计算序列平均 value
values_sum = (values * response_mask).sum(dim=-1)  # (batch,)
values_count = response_mask.sum(dim=-1).clamp(min=1)  # (batch,)
sequence_value = values_sum / values_count  # (batch,)

# 为了保持接口一致（后续代码期望 shape 为 (batch, seq_len)）
# 将平均值放在最后一个有效位置，其他位置为 0
values_output = torch.zeros_like(values)
last_indices = (response_mask.sum(dim=-1) - 1).long()
batch_indices = torch.arange(values.size(0), device=values.device)
values_output[batch_indices, last_indices] = sequence_value

return values_output
```

### 关键改进

1. **计算所有 token 的平均 value**
   ```python
   sequence_value = sum(values) / count(values)
   ```

2. **梯度反向传播到所有 token**
   ```python
   ∂loss/∂values[i] = (1/N) * ∂loss/∂sequence_value  # 所有 i
   ```

3. **保持接口兼容**
   - 输出 shape 仍然是 `(batch, seq_len)`
   - 平均值放在最后一个有效位置
   - 后续代码无需修改

## 为什么平均值有效？

### 1. 梯度反向传播机制

```python
# Last Token
score = values[last]
loss.backward()
→ 只有 values[last] 有梯度
→ 模型只更新最后一个 token 相关的参数
→ 不需要"阅读"前面的内容

# 平均值
score = mean(values)
loss.backward()
→ 所有 values[i] 都有梯度
→ 模型必须更新所有 token 相关的参数
→ 被迫"阅读"整个序列
```

### 2. 强制语义理解

```python
# 平均值 = 所有 token 的贡献之和
score = (v0 + v1 + ... + vN) / N

# 如果某个 token 的 value 很差
→ 会拉低整体 score
→ 模型必须学习让每个 token 的 value 都合理
→ 必须理解每个 token 的语义贡献
```

### 3. 自然的语义聚合

```python
# 好的回答
"网络年夜饭、朋友年夜饭。"
→ 每个词都有意义
→ 每个 token 的 value 都较高
→ 平均值高

# 差的回答
"啊啊啊啊啊啊啊啊啊。"
→ 大部分词无意义
→ 大部分 token 的 value 低
→ 平均值低
```

## 预期效果

### 修改前 (Last Token)

```python
案例 1:
Student (逗号): -0.56
Teacher (句号):  2.50
差异: 3.06 (不合理)

案例 2:
Student (完整回答): -2.33
Teacher (简洁回答):  2.84
差异: 5.17 (完全不合理)

问题:
- Critic 依赖标点符号/位置
- 完全忽略内容语义
- 随机的评分
```

### 修改后 (平均值)

```python
案例 1:
Student (逗号): 1.8 ~ 2.0
Teacher (句号): 1.9 ~ 2.1
差异: 0.1 ~ 0.3 (合理)

案例 2:
Student (完整回答): 2.0 ~ 2.5
Teacher (简洁回答): 2.3 ~ 2.8
差异: 0.3 ~ 0.5 (合理)

改进:
- Critic 考虑整体内容
- 理解语义质量
- 合理的评分
```

## 实施步骤

### 1. 代码已修改

✅ `verl/verl/workers/critic/dp_critic.py` 已更新
- `_forward_micro_batch` 方法改用平均值
- 同时支持 `use_remove_padding=True/False` 两种模式

### 2. 需要重新训练

⚠️ **必须重新运行 Warmup 阶段**

理由：
- 当前 Critic 是用 Last Token 训练的
- 学到的是错误的模式（标点符号/位置）
- 必须用平均值重新训练，学习真正的语义

### 3. 重新训练步骤

```bash
# 1. 清理旧的 checkpoint（可选，但推荐）
rm -rf outputs/warmup_checkpoints/*

# 2. 重新运行 Warmup
bash scripts/train/A3b_gspo/content_merge_trainning/A3b-warmup-gspo-optimized.sh

# 3. 观察训练日志
tail -f logs/critic_scoring_details/*.log
```

### 4. 验证效果

**检查点**:

```python
# 观察打分详情日志
# 应该看到：

✅ 相似内容得到相似分数
  Student: "网络年夜饭、朋友年夜饭" → 2.0
  Teacher: "网络年夜饭、朋友年夜饭" → 2.1
  差异: 0.1 (合理)

✅ 完整回答不会得负分
  Student: "根据文章，...包括网络年夜饭和朋友年夜饭。" → 1.8
  Teacher: "网络年夜饭、朋友年夜饭。" → 2.0
  差异: 0.2 (合理，teacher 更简洁)

✅ 标点符号差异不会导致巨大分差
  Student: "...，体重..." → 1.9
  Teacher: "...。体重..." → 2.0
  差异: 0.1 (合理)
```

## 技术细节

### 接口兼容性

**问题**: 后续代码期望 `values` 的 shape 为 `(batch, seq_len)`

**解决**: 将平均值放在最后一个有效位置

```python
# 计算平均值
sequence_value = values.mean()  # scalar

# 创建输出 tensor
values_output = torch.zeros_like(values)  # (batch, seq_len)

# 将平均值放在最后一个有效位置
last_indices = (response_mask.sum(dim=-1) - 1).long()
values_output[batch_indices, last_indices] = sequence_value

# 后续代码
score = values_output.sum(dim=-1)  # 提取平均值
# 因为只有最后一个位置有值，sum 就是平均值
```

### 梯度流向

```python
# 前向传播
values = model(input)  # (batch, seq_len)
sequence_value = (values * mask).sum() / mask.sum()
values_output[last_pos] = sequence_value
score = values_output.sum()
loss = discriminator_loss(score)

# 反向传播
∂loss/∂score = g
∂score/∂values_output[last_pos] = 1
∂values_output[last_pos]/∂sequence_value = 1
∂sequence_value/∂values[i] = mask[i] / mask.sum()  # 所有 i

# 最终梯度
∂loss/∂values[i] = g * mask[i] / mask.sum()  # 所有 i 都有梯度！
```

## 风险与缓解

### 风险 1: 训练时间增加

**风险**: 需要重新 Warmup 253 步

**缓解**: 
- 这是必要的，当前 Critic 是错误的
- 253 步相对较快（约 12 小时）
- 避免了 Stage 2 的失败（节省更多时间）

### 风险 2: 可能需要调整超参数

**风险**: 平均值可能改变分数分布

**缓解**:
- 当前超参数（temperature=5.0）应该仍然适用
- 如果需要，可以微调 temperature
- 观察前 50 步，必要时调整

### 风险 3: 接口兼容性

**风险**: 后续代码可能依赖 Last Token 机制

**缓解**:
- 已保持输出 shape 一致
- 平均值放在最后一个位置
- 后续代码无需修改

## 监控指标

### 重新训练时关注

```python
# 核心指标
critic/d_acc: 70-85%  # 应该在合理范围
critic/score_diff: 1-3  # 应该比之前小（更合理的区分）
critic/score_separation: 1-2  # 区分度

# 新的期望
相同内容的分差: < 0.5  # 应该很小
完整回答的分数: > 0  # 不应该是负数
标点差异的分差: < 0.3  # 应该很小
```

### 健康训练的特征

```python
✅ 相似内容得到相似分数
✅ 完整回答得到合理分数（不是负数）
✅ 标点符号差异不会导致巨大分差
✅ d_acc 在 70-85% 范围
✅ 训练稳定，无梯度爆炸
```

## Critic 打分机制深度解析

### Critic 模型的完整架构

#### 1. 模型结构

```python
# Critic = Transformer + Value Head
class Critic:
    def __init__(self):
        self.transformer = PretrainedModel()  # 如 Qwen3-14B
        self.value_head = nn.Linear(hidden_dim, 1)  # 线性投影层
    
    def forward(self, input_ids):
        # Step 1: Transformer 编码
        hidden_states = self.transformer(input_ids)
        # 输出: (batch, seq_len, hidden_dim)
        # 每个 token 都有一个 hidden state
        
        # Step 2: Value Head 对每个 hidden state 打分
        values = self.value_head(hidden_states)
        # 输出: (batch, seq_len, 1) → squeeze → (batch, seq_len)
        # 每个 token 都有一个 value 分数！
        
        return values
```

#### 2. 每个 Token 都有自己的 Value

```python
# 输入
Input: [Prompt tokens, Response tokens]
       [t1, t2, ..., tN, r1, r2, ..., rM]

# Transformer 处理（因果注意力）
h1 = f(t1)                    # t1 的 hidden state
h2 = f(t1, t2)                # t2 的 hidden state（可以看到 t1）
h3 = f(t1, t2, t3)            # t3 的 hidden state（可以看到 t1, t2）
...
hM = f(t1, ..., tN, r1, ..., rM)  # rM 的 hidden state（可以看到所有）

# Value Head 对每个 hidden state 独立打分
v1 = value_head(h1)  # t1 的 value
v2 = value_head(h2)  # t2 的 value
v3 = value_head(h3)  # t3 的 value
...
vM = value_head(hM)  # rM 的 value

# 关键点
每个 token 都有一个独立的 value 分数！
```

#### 3. 每个 Value 的含义

```python
# 例子
Response: "网络年夜饭、朋友年夜饭。"
Tokens:   [t1="网络", t2="年夜饭", t3="、", t4="朋友", t5="年夜饭", t6="。"]

# 每个 token 的 value 含义
v1: 基于 [prompt, "网络"] 的评分
v2: 基于 [prompt, "网络", "年夜饭"] 的评分
v3: 基于 [prompt, "网络", "年夜饭", "、"] 的评分
v4: 基于 [prompt, ..., "朋友"] 的评分
v5: 基于 [prompt, ..., "年夜饭"] 的评分
v6: 基于 [prompt, ..., "。"] 的评分（完整序列）

# 每个 value 反映了"到该 token 为止"的序列质量
```

### Last Token vs 平均值的本质区别

#### Last Token 方法（旧）

```python
# 只使用最后一个 token 的 value
score = values[last_token]  # 只用 v6

# 问题
1. 忽略了 v1, v2, ..., v5
2. 梯度只反向传播到 v6
   ∂loss/∂v6 = g  ✅ 有梯度
   ∂loss/∂v1 = 0  ❌ 没有梯度
   ∂loss/∂v2 = 0  ❌ 没有梯度
   ...

3. 只有 h6 相关的参数被更新
4. 前面 token 的参数不更新
5. 虽然 h6 通过 attention 看到了所有 token
   但梯度不会反向传播到前面的 token
```

#### 平均值方法（新）

```python
# 使用所有 token 的 value 的平均
score = mean([v1, v2, v3, v4, v5, v6])
      = (v1 + v2 + v3 + v4 + v5 + v6) / 6

# 优势
1. 使用了所有 token 的 value
2. 梯度反向传播到所有 token
   ∂loss/∂v1 = g/6  ✅ 有梯度
   ∂loss/∂v2 = g/6  ✅ 有梯度
   ...
   ∂loss/∂v6 = g/6  ✅ 有梯度

3. 所有 hidden states 相关的参数都被更新
4. 模型被迫学习每个 token 的贡献
5. 更强的学习信号
```

### 梯度反向传播详解

#### 完整的梯度流

```python
# 前向传播
input_ids → transformer → hidden_states → value_head → values
[t1, ..., t6] → [h1, ..., h6] → [v1, ..., v6]

# 平均值
score = (v1 + v2 + v3 + v4 + v5 + v6) / 6

# 损失
loss = discriminator_loss(score)

# 反向传播
∂loss/∂score = g

# 平均值的梯度
∂score/∂v1 = 1/6
∂score/∂v2 = 1/6
...
∂score/∂v6 = 1/6

# 链式法则
∂loss/∂v1 = ∂loss/∂score * ∂score/∂v1 = g * (1/6)
∂loss/∂v2 = g * (1/6)
...
∂loss/∂v6 = g * (1/6)

# 继续反向传播到 hidden states
∂loss/∂h1 = ∂loss/∂v1 * ∂v1/∂h1 = g * (1/6) * W_value
∂loss/∂h2 = g * (1/6) * W_value
...
∂loss/∂h6 = g * (1/6) * W_value

# 最终反向传播到 transformer 参数
所有 token 相关的 transformer 参数都会被更新！
```

#### 为什么这很重要？

```python
# Last Token 方法
- 只有最后一个 token 的路径有梯度
- 前面 token 的参数不更新
- 模型不需要"理解"前面的内容
- 只需要让最后一个 token 的 value 合适即可

# 平均值方法
- 所有 token 的路径都有梯度
- 所有 token 的参数都更新
- 模型必须"理解"每个 token 的贡献
- 被迫学习整个序列的语义
```

### 实际训练效果对比

#### 未训练模型（当前状态）

```python
# Last Token 方法
Response: "根据文章，因为各种原因无法回家团聚的人的年夜饭方式正在逐渐改变，包括网络年夜饭和朋友年夜饭。"
v_last = -2.33  # 最后一个 token 的随机 value

问题:
- 单个 token 的 value 方差很大（σ²）
- 分数不稳定，随机性强
- 完全依赖最后一个 token 的随机初始化

# 平均值方法
Response: "根据文章，因为各种原因无法回家团聚的人的年夜饭方式正在逐渐改变，包括网络年夜饭和朋友年夜饭。"
values = [0.5, -0.3, 1.2, 0.8, -0.5, ..., 0.2, -0.1]  # 29 个 token 的 value
mean(values) = 0.15  # 平均后

优势:
- 平均 N 个 token，方差减小到 σ²/N
- 分数更稳定
- 不依赖单个 token 的随机值
```

#### 训练后（预期效果）

```python
# 模型学习后
Response: "网络年夜饭、朋友年夜饭。"

# 每个 token 的 value 会反映其语义贡献
v1 = 0.8   # "网络" - 有意义的词，中等贡献
v2 = 1.2   # "年夜饭" - 关键词，高贡献
v3 = 0.1   # "、" - 标点符号，低贡献
v4 = 0.9   # "朋友" - 有意义的词，中等贡献
v5 = 1.3   # "年夜饭" - 关键词，高贡献
v6 = 0.2   # "。" - 标点符号，低贡献

mean([0.8, 1.2, 0.1, 0.9, 1.3, 0.2]) = 0.75

# 模型学到了
- 关键词（年夜饭）→ 高 value
- 有意义的词（网络、朋友）→ 中等 value
- 标点符号 → 低 value
- 平均值综合反映了整体质量

# 对比差的回答
Response: "啊啊啊啊啊啊啊啊啊。"
values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
mean(values) = 0.11  # 明显更低

# 模型真正学会了区分内容质量！
```

### 原始 GAD 设计的理论分析

#### Last Token 的理论基础

```python
# 原始 GAD 设计的合理性
1. hidden_state[last_token] 通过 attention 包含了所有 token 的信息
2. 理论上，value_head(hidden_state[last_token]) 可以评估整个序列
3. 这是一个合理的序列级评分方法

# 为什么在实践中失败？
问题: 未训练的基础模型
- 基础模型（Qwen3-14B）预训练目标是语言建模
- 没有训练过"序列评分"任务
- hidden_state[last_token] 虽然包含信息，但不知道如何评分
- value_head 随机初始化，输出随机分数

结果:
hidden_state[last_token] → value_head → 随机分数
```

#### 平均值的优势

```python
# 为什么平均值在未训练模型上更好？

1. 减小方差（统计学优势）
   Var(v_last) = σ²
   Var(mean(v)) = σ²/N
   → 在未训练模型中，σ² 很大，平均值更稳定

2. 更强的梯度信号（优化优势）
   Last Token: 只有 1 个 token 有梯度
   平均值: 所有 N 个 token 都有梯度
   → 更快的收敛

3. 隐式正则化（泛化优势）
   Last Token: 可能过度依赖某个 token
   平均值: 强制考虑所有 token
   → 更好的泛化能力
```

#### 长期考虑：渐进式策略

```python
# 理论上，Last Token 可能有更高的性能上限
原因:
- 非线性聚合，表达能力更强
- 可以学习复杂的模式

# 但需要充分训练
问题:
- 在未训练模型上完全随机
- 需要大量训练才能收敛

# 可能的最优策略
阶段 1: Warmup (0-253 steps)
  score = mean(values)  # 平均值，稳定训练

阶段 2: 过渡 (254-500 steps)
  α = (step - 254) / 246
  score = (1-α) * mean(values) + α * values[last]  # 渐进过渡

阶段 3: 成熟 (500+ steps)
  score = values[last]  # Last Token，最大表达能力

# 当前建议
先用平均值完成 Warmup，观察效果
如果需要更高性能，再考虑渐进式策略
```

### EOS Token 处理

#### 自动排除机制

```python
# 平均值方法自动处理 EOS Token
response_mask = attention_mask[:, -response_length:]
# response_mask 中，EOS 之后的位置都是 0

values_sum = (values * response_mask).sum(dim=-1)
# mask=0 的位置，values 被乘以 0，不参与求和

values_count = response_mask.sum(dim=-1)
# 只计算 mask=1 的位置数量

sequence_value = values_sum / values_count
# 平均值 = 有效 token 的 value 之和 / 有效 token 数量

# 例子
Student: [token1, token2, EOS, PAD, PAD]
  mask:  [1,      1,      0,   0,   0]
  values: [v1,    v2,     v3,  v4,  v5]
  
  values_sum = v1*1 + v2*1 + v3*0 + v4*0 + v5*0 = v1 + v2
  values_count = 1 + 1 + 0 + 0 + 0 = 2
  sequence_value = (v1 + v2) / 2

Teacher: [token1, token2, PAD, PAD, PAD]
  mask:  [1,      1,      0,   0,   0]
  values: [v1,    v2,     v3,  v4,  v5]
  
  values_sum = v1 + v2
  values_count = 2
  sequence_value = (v1 + v2) / 2

# 结果：两者计算的是相同范围的平均值
# ✅ EOS 被自动排除
# ✅ 相同内容得到相同分数
```

## 总结

### 核心改进

1. **从 Last Token 改为平均值**
   - 使用所有 token 的 value，而非只用一个
   - 梯度反向传播到所有 token
   - 强制模型学习整个序列的语义

2. **解决了根本性缺陷**
   - Critic 不再依赖标点符号/位置
   - 真正理解内容质量
   - 合理的评分

3. **保持接口兼容**
   - 输出 shape 不变
   - 后续代码无需修改

### 关键洞察

1. **Critic 对每个 Token 都打分**
   - Value Head 对每个 hidden state 独立应用
   - 每个 token 都有一个 value 分数
   - 不是"不打分"，而是"如何使用这些分数"

2. **平均值更充分利用了打分信息**
   - Last Token: 只用 v_last，忽略其他
   - 平均值: 用所有 v1, v2, ..., vN
   - 更充分地利用了 Critic 的打分能力

3. **梯度反向传播是关键**
   - 平均值让所有 token 都有梯度
   - 模型被迫学习每个 token 的语义贡献
   - 这是平均值有效的根本原因

### 理论基础

1. **统计学优势**: 方差减小（σ²/N）
2. **优化优势**: 更强的梯度信号
3. **泛化优势**: 隐式正则化

### 下一步

1. ✅ 代码已修改
2. ⏳ 重新运行 Warmup (253 步)
3. ⏳ 验证评分是否合理
4. ⏳ 确认后进入 Stage 2

### 预期收益

- Critic 真正理解语义
- 合理的评分机制
- Stage 2 (GAD) 训练成功
- Actor 学到正确的策略

---

**修改日期**: 2026-01-28
**修改者**: AI Assistant
**状态**: ✅ 代码已修改，待重新训练验证
**最后更新**: 2026-01-28 (添加深度技术分析)
