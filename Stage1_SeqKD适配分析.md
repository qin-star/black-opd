# Stage 1 (SeqKD) 适配分析与修改方案

## 一、SeqKD 阶段概述

### 1.1 核心特点

**SeqKD (Sequence Knowledge Distillation)** 是 GAD 训练的第一阶段，本质上是**使用 GRPO 基础设施的纯 SFT 训练**。

**关键特征**：
- ✅ 生成 8 个响应用于监控质量（Rouge-L）
- ✅ 训练时只使用 `teacher_response`，丢弃 8 个采样
- ✅ 使用 SFT 损失：`-mean(log P(teacher_response))`
- ❌ 不使用 GRPO 优势计算
- ❌ 不使用强化学习

### 1.2 训练流程

```
数据加载 (32 prompts + teacher_response)
  ↓
VLLM 生成 (n=8, 共 256 个响应) → 用于 Rouge-L 监控
  ↓
数据扩展 (repeat n=8, 共 256 个样本)
  ↓
选择 teacher 数据 (丢弃 8 个采样的 responses)
  ↓
SFT 训练 (Loss = -mean(log P(teacher_response)))
```

---

## 二、与 Warmup/GAD 阶段的区别

| 维度 | Stage 1 (SeqKD) | Stage 2 (Warmup) | Stage 3 (GAD) |
|------|----------------|------------------|---------------|
| **训练目标** | 模仿 teacher | 训练判别器 + Actor | 对抗训练 |
| **损失函数** | SFT 损失 | 判别器损失 + PPO | 判别器损失 + GSPO |
| **使用采样** | ❌ 只用 teacher | ❌ 只用 teacher | ✅ 使用 8 个采样 |
| **Critic 作用** | ❌ 不使用 | ✅ 判别器 | ✅ 判别器 |
| **优势估计** | ❌ 不使用 | ✅ GRPO | ✅ GRPO |
| **Critic Warmup** | N/A | 10 步 | 0 步 |

---

## 三、新框架的现状分析

### 3.1 已完成的功能（Warmup/GAD 适配）

从之前的修改中，我们已经完成：
- ✅ `dp_critic.py`：判别器训练
- ✅ `rl_dataset.py`：`teacher_response` 加载
- ✅ `core_algos.py`：`compute_discriminator_loss`
- ✅ GRPO 优势估计（框架内置）
- ✅ GSPO 策略损失（框架内置）

### 3.2 SeqKD 阶段缺失的功能

经过检查，新框架**缺少以下 SeqKD 特定功能**：

#### ❌ 1. `compute_sft_loss` 函数

**旧框架位置**：`verl/trainer/ppo/core_algos.py`

**功能**：
```python
def compute_sft_loss(log_prob, response_mask):
    """
    计算监督微调损失
    Loss = -mean(log_prob * mask)
    """
    sft_loss = -masked_mean(log_prob, response_mask)
    return sft_loss
```

**状态**：❌ **新框架中不存在**

---

#### ❌ 2. Actor 的 SFT 训练模式

**旧框架位置**：`verl/workers/actor/dp_actor.py` 的 `update_policy` 方法

**功能**：
```python
def update_policy(self, data):
    # 选择 teacher 数据
    select_keys = [
        "teacher_response",
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_position_ids"
    ]
    # 注意：丢弃 responses（8 个采样）
    
    # 前向传播使用 teacher 数据
    log_prob = self._forward_micro_batch(teacher_data)
    
    # 计算 SFT 损失
    sft_loss = compute_sft_loss(log_prob, teacher_response_mask)
    
    # 反向传播
    sft_loss.backward()
```

**状态**：❌ **新框架的 Actor 不支持 teacher 数据训练**

---

#### ❌ 3. 数据选择逻辑

**旧框架逻辑**：
```python
# 在 update_policy 中
if use_sft_mode:  # SeqKD 阶段
    select_keys = ["teacher_response", "teacher_input_ids", ...]
    # 丢弃 responses
else:  # Warmup/GAD 阶段
    select_keys = ["responses", "input_ids", ...]
    # 使用学生响应
```

**状态**：❌ **新框架没有这个分支逻辑**

---

#### ✅ 4. Rouge-L 评估（可能已存在）

**功能**：在验证阶段计算 Rouge-L 分数

**状态**：❓ **需要检查，可能已在框架中**

---

## 四、需要添加的修改

### 🔴 必须修改（核心功能）

#### 修改 1：添加 `compute_sft_loss` 函数

**文件**：`verl/trainer/ppo/core_algos.py`

**位置**：在 `compute_discriminator_loss` 之后

**代码**：
```python
def compute_sft_loss(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """
    Compute supervised fine-tuning loss for SeqKD stage.
    
    Args:
        log_prob (torch.Tensor):
            Log probabilities of tokens, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask for valid response tokens, shape (batch_size, response_length).
        loss_agg_mode (str):
            Loss aggregation mode. Defaults to "token-mean".
    
    Returns:
        sft_loss (torch.Tensor):
            Scalar SFT loss.
    """
    # SFT loss: maximize log probability of teacher response
    # Equivalent to minimizing negative log likelihood
    sft_loss = -agg_loss(loss_mat=log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    return sft_loss
```

---

#### 修改 2：修改 Actor 的 `update_policy` 方法

**文件**：`verl/workers/actor/dp_actor.py`

**修改位置**：`update_policy` 方法的开头

**修改内容**：添加 SFT 模式支持

```python
def update_policy(self, data: DataProto):
    """Update the policy network using PPO or SFT."""
    self.actor_module.train()
    
    # Check if using SFT mode (SeqKD stage)
    use_sft_mode = data.meta_info.get("use_sft_mode", False)
    
    if use_sft_mode:
        # SeqKD stage: use teacher data only
        select_keys = [
            "teacher_response",
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
        ]
    else:
        # Warmup/GAD stage: use student responses
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
    
    # ... 继续原有逻辑 ...
```

**修改位置 2**：前向传播部分

```python
# 在前向传播时
if use_sft_mode:
    # Use teacher data
    input_ids = model_inputs["teacher_input_ids"]
    attention_mask = model_inputs["teacher_attention_mask"]
    position_ids = model_inputs["teacher_position_ids"]
    response_length = model_inputs["teacher_response"].size(-1)
else:
    # Use student data
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    position_ids = model_inputs["position_ids"]
    response_length = model_inputs["responses"].size(-1)

# 前向传播
output = self.actor_module(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
)
```

**修改位置 3**：损失计算部分

```python
# 计算 log_prob
logits = output.logits
log_prob = compute_log_prob(logits, input_ids, response_length)

if use_sft_mode:
    # SeqKD stage: use SFT loss
    response_mask = attention_mask[:, -response_length:]
    policy_loss = compute_sft_loss(
        log_prob=log_prob,
        response_mask=response_mask,
        loss_agg_mode=self.config.loss_agg_mode,
    )
    
    micro_batch_metrics = {
        "actor/sft_loss": policy_loss.detach().item(),
        "actor/teacher_pg_loss": policy_loss.detach().item(),  # 兼容旧日志
    }
else:
    # Warmup/GAD stage: use PPO/GSPO loss
    old_log_prob = model_inputs["old_log_probs"]
    advantages = model_inputs["advantages"]
    
    policy_loss, pg_metrics = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ...
    )
    
    micro_batch_metrics = pg_metrics
```

---

#### 修改 3：修改训练脚本配置

**文件**：启动脚本（如 `gpt5-8b-seqkd.sh`）

**添加配置**：
```bash
# SeqKD 阶段特定配置
+actor_rollout_ref.actor.use_sft_mode=true  # 启用 SFT 模式
trainer.critic_warmup=-1  # 不使用 Critic（或设置很大的值）
```

---

### 🟡 可选修改（增强功能）

#### 修改 4：添加 Rouge-L 评估（如果不存在）

**文件**：`verl/trainer/ppo/ray_trainer.py`

**位置**：验证循环中

**功能**：
```python
def validate(self, dataloader):
    # ... 生成响应 ...
    
    # 计算 Rouge-L 分数
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for gen_response, teacher_response in zip(generated, teachers):
        score = scorer.score(teacher_response, gen_response)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    metrics['val/rouge-L/mean'] = np.mean(rouge_scores)
    
    return metrics
```

---

## 五、修改优先级

### 🔴 高优先级（必须完成）

1. **添加 `compute_sft_loss` 函数**
   - 位置：`core_algos.py`
   - 难度：⭐ 简单
   - 影响：核心功能

2. **修改 Actor 的 `update_policy` 方法**
   - 位置：`dp_actor.py`
   - 难度：⭐⭐⭐ 中等
   - 影响：核心功能

3. **添加 `use_sft_mode` 配置支持**
   - 位置：配置文件和 `meta_info`
   - 难度：⭐ 简单
   - 影响：核心功能

### 🟡 中优先级（建议完成）

4. **添加 Rouge-L 评估**
   - 位置：`ray_trainer.py`
   - 难度：⭐⭐ 简单
   - 影响：监控质量

### 🟢 低优先级（可选）

5. **优化日志输出**
   - 添加 `actor/sft_loss` 指标
   - 添加 `actor/teacher_pg_loss` 指标（兼容性）

---

## 六、完整的修改方案

### 方案 A：最小修改（推荐）

**只修改 Actor，不添加新函数**

```python
# 在 dp_actor.py 的 update_policy 中
if use_sft_mode:
    # 直接使用负 log_prob 作为损失
    policy_loss = -masked_mean(log_prob, response_mask)
else:
    # 使用 PPO/GSPO 损失
    policy_loss = compute_policy_loss(...)
```

**优点**：
- ✅ 修改最小
- ✅ 不需要添加新函数
- ✅ 逻辑清晰

**缺点**：
- ❌ 代码重复
- ❌ 不够模块化

---

### 方案 B：完整修改（标准）

**添加 `compute_sft_loss` 函数 + 修改 Actor**

```python
# core_algos.py
def compute_sft_loss(log_prob, response_mask, loss_agg_mode="token-mean"):
    return -agg_loss(loss_mat=log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

# dp_actor.py
if use_sft_mode:
    policy_loss = compute_sft_loss(log_prob, response_mask)
else:
    policy_loss = compute_policy_loss(...)
```

**优点**：
- ✅ 代码模块化
- ✅ 易于维护
- ✅ 与旧框架一致

**缺点**：
- ❌ 需要修改两个文件

---

## 七、数据流验证

### 7.1 SeqKD 阶段的数据流

```python
# 1. 数据加载（rl_dataset.py）
{
    "content": [...],
    "teacher_response": "教师回复",  # ✅ 已支持
    "teacher_input_ids": [...],     # ✅ 已支持
    "teacher_attention_mask": [...], # ✅ 已支持
    "teacher_position_ids": [...],   # ✅ 已支持
}

# 2. Rollout 生成（可选，用于监控）
gen_batch = actor_rollout_wg.generate_sequences(batch)  # 生成 8 个响应
# 计算 Rouge-L（仅监控）

# 3. 数据扩展
batch = batch.repeat(n=8)  # 256 个样本

# 4. Actor 训练
batch.meta_info["use_sft_mode"] = True  # ← 关键：设置 SFT 模式
actor_output = actor_rollout_wg.update_actor(batch)
```

### 7.2 需要验证的点

- ✅ `teacher_response` 是否正确加载（已完成）
- ✅ `teacher_input_ids` 等字段是否正确构建（已完成）
- ❓ `meta_info["use_sft_mode"]` 是否正确传递
- ❓ Actor 是否正确使用 teacher 数据

---

## 八、配置文件示例

### SeqKD 阶段配置

```yaml
# 算法配置
algorithm:
  adv_estimator: grpo  # 使用 GRPO 框架（但不用优势计算）

# 数据配置
data:
  train_files: /path/to/data_with_teacher.parquet
  train_batch_size: 256  # 32 prompts × 8
  max_prompt_length: 2048
  max_response_length: 1536

# Actor 配置
actor_rollout_ref:
  model:
    path: /path/to/base/model
  
  actor:
    use_sft_mode: true  # 🔥 启用 SFT 模式
    optim.lr: 5e-6
    ppo_mini_batch_size: 256
    use_dynamic_bsz: true
  
  rollout:
    n: 8  # 生成 8 个响应（用于监控）
    temperature: 0.8

# Trainer 配置
trainer:
  critic_warmup: -1  # 不使用 Critic（或设置很大的值）
  total_epochs: 4
  save_freq: 50
  test_freq: 50
```

---

## 九、与其他阶段的兼容性

### 9.1 向后兼容

**Warmup/GAD 阶段不受影响**：
```yaml
actor_rollout_ref.actor.use_sft_mode: false  # 或不设置（默认 false）
```

### 9.2 检查点兼容

**SeqKD → Warmup**：
```bash
# SeqKD 训练
bash seqkd_script.sh --exp_name seqkd_exp

# 从 SeqKD 检查点继续 Warmup 训练
bash warmup_script.sh \
  --model /path/to/seqkd/checkpoint/actor \
  --reward_model /path/to/reward_model \
  --exp_name warmup_exp \
  actor_rollout_ref.actor.use_sft_mode=false  # 关闭 SFT 模式
```

---

## 十、测试验证

### 10.1 单元测试

```python
# 测试 compute_sft_loss
def test_compute_sft_loss():
    log_prob = torch.randn(4, 10)  # (batch, seq_len)
    response_mask = torch.ones(4, 10)
    
    loss = compute_sft_loss(log_prob, response_mask)
    
    assert loss.dim() == 0  # 标量
    assert loss < 0  # 负 log prob
```

### 10.2 集成测试

```bash
# 1. 运行小规模 SeqKD 训练
bash seqkd_script.sh \
  --model /path/to/model \
  --exp_name seqkd_test \
  trainer.total_epochs=1

# 2. 检查日志指标
# - actor/sft_loss 应该存在
# - actor/teacher_pg_loss 应该存在
# - val/rouge-L/mean 应该存在（如果实现）

# 3. 检查检查点
# - 应该能正常保存和加载
```

---

## 十一、总结

### 11.1 核心修改

**必须完成的 2 个修改**：
1. ✅ 添加 `compute_sft_loss` 函数到 `core_algos.py`
2. ✅ 修改 `dp_actor.py` 的 `update_policy` 方法，支持 `use_sft_mode`

### 11.2 修改影响

| 修改 | 影响范围 | 风险 |
|------|---------|------|
| `compute_sft_loss` | 新增函数 | ✅ 低（不影响现有功能） |
| `update_policy` | Actor 训练逻辑 | ⚠️ 中（需要仔细测试） |
| 配置参数 | 启动脚本 | ✅ 低（向后兼容） |

### 11.3 与已完成修改的关系

**已完成的修改（Warmup/GAD）**：
- ✅ `dp_critic.py`：判别器训练
- ✅ `rl_dataset.py`：teacher 数据加载
- ✅ `core_algos.py`：判别器损失

**新增修改（SeqKD）**：
- 🆕 `core_algos.py`：SFT 损失
- 🆕 `dp_actor.py`：SFT 训练模式

**关系**：
- ✅ 互不冲突
- ✅ 共享数据加载逻辑
- ✅ 通过配置切换模式

### 11.4 推荐的实现顺序

1. **第一步**：添加 `compute_sft_loss` 函数（5 分钟）
2. **第二步**：修改 `dp_actor.py` 的 `update_policy` 方法（30 分钟）
3. **第三步**：创建 SeqKD 启动脚本（10 分钟）
4. **第四步**：运行测试验证（1 小时）
5. **第五步**（可选）：添加 Rouge-L 评估（30 分钟）

**总计**：约 2-3 小时完成核心功能

---

## 十二、下一步行动

### 立即可以做的

1. ✅ **添加 `compute_sft_loss` 函数**
   - 简单直接
   - 不影响现有功能

2. ✅ **修改 `dp_actor.py`**
   - 添加 `use_sft_mode` 分支
   - 支持 teacher 数据训练

3. ✅ **创建启动脚本**
   - 基于现有脚本修改
   - 添加 `use_sft_mode=true`

### 如果需要帮助

我可以：
1. 提供完整的代码实现
2. 帮助调试错误
3. 优化训练配置
4. 添加监控指标

---

**最终结论**：SeqKD 阶段需要 **2 个核心修改**，修改量不大，与已完成的 Warmup/GAD 修改互不冲突。
