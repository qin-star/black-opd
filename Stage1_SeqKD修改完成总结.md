# Stage 1 (SeqKD) 修改完成总结

## ✅ 修改完成！

我已经成功完成了 Stage 1 (SeqKD) 阶段所需的所有代码修改。

---

## 一、已完成的修改

### 修改 1：添加 `compute_sft_loss` 函数 ✅

**文件**：`verl/trainer/ppo/core_algos.py`

**位置**：第 1478-1505 行（在 `compute_discriminator_loss` 之后）

**功能**：
```python
def compute_sft_loss(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """
    Compute supervised fine-tuning loss for SeqKD stage.
    Loss = -mean(log_prob * mask)
    """
    sft_loss = -agg_loss(loss_mat=log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return sft_loss
```

---

### 修改 2：修改 Actor 的 `update_policy` 方法 ✅

**文件**：`verl/workers/actor/dp_actor.py`

**修改内容**：

#### 2.1 添加 SFT 模式检测（第 377-400 行）

```python
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
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "advantages",
    ]
```

#### 2.2 添加 teacher 数据处理（第 439-448 行）

```python
if use_sft_mode:
    # SeqKD stage: use teacher data
    response_length = model_inputs["teacher_response"].size(-1)
    teacher_attention_mask = model_inputs["teacher_attention_mask"]
    response_mask = teacher_attention_mask[:, -response_length:]
else:
    # Warmup/GAD stage: use student data
    response_mask = model_inputs["response_mask"]
    old_log_prob = model_inputs["old_log_probs"]
    advantages = model_inputs["advantages"]
```

#### 2.3 添加 teacher 前向传播（第 460-494 行）

```python
if use_sft_mode:
    # SeqKD stage: forward pass with teacher data
    model_inputs["input_ids"] = model_inputs["teacher_input_ids"]
    model_inputs["attention_mask"] = model_inputs["teacher_attention_mask"]
    model_inputs["position_ids"] = model_inputs["teacher_position_ids"]
    
    entropy, log_prob = self._forward_micro_batch(
        model_inputs, temperature=temperature, calculate_entropy=False
    )
else:
    # Warmup/GAD stage: forward pass with student data
    entropy, log_prob = self._forward_micro_batch(
        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
    )
```

#### 2.4 添加 SFT 损失计算（第 502-532 行）

```python
if use_sft_mode:
    # SeqKD stage: use SFT loss
    from verl.trainer.ppo.core_algos import compute_sft_loss
    
    pg_loss = compute_sft_loss(
        log_prob=log_prob,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    
    micro_batch_metrics.update({
        "actor/sft_loss": pg_loss.detach().item(),
        "actor/teacher_pg_loss": pg_loss.detach().item(),  # For compatibility
    })
else:
    # Warmup/GAD stage: use PPO/GSPO loss
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(...)
    micro_batch_metrics.update(pg_metrics)
```

---

## 二、修改特点

### ✅ 向后兼容

- **不影响 Warmup/GAD 阶段**：通过 `use_sft_mode` 标志控制
- **默认行为不变**：`use_sft_mode=False` 时使用原有逻辑
- **配置切换简单**：只需设置 `meta_info["use_sft_mode"]=True`

### ✅ 代码简洁

- **无冗余代码**：复用现有的 `_forward_micro_batch` 方法
- **逻辑清晰**：通过 `if use_sft_mode` 分支明确区分两种模式
- **易于维护**：所有 SFT 相关逻辑集中在一处

### ✅ 功能完整

- **支持 teacher 数据训练**：正确处理 `teacher_input_ids` 等字段
- **SFT 损失计算**：使用负对数似然损失
- **指标输出**：`actor/sft_loss` 和 `actor/teacher_pg_loss`

---

## 三、使用方法

### 3.1 数据准备

**数据格式**（与 Warmup/GAD 相同）：
```python
{
    "content": [{"role": "user", "content": "问题"}],
    "teacher_response": "教师模型的高质量回复"  # 必须包含
}
```

**数据加载**：已在 `rl_dataset.py` 中完成，无需额外修改

---

### 3.2 配置文件

**SeqKD 阶段配置**：

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
    optim.lr: 5e-6
    ppo_mini_batch_size: 256
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 20480
  
  rollout:
    n: 8  # 生成 8 个响应（用于监控）
    temperature: 0.8

# Trainer 配置
trainer:
  critic_warmup: -1  # 不使用 Critic（设置为 -1 或很大的值）
  total_epochs: 4
  save_freq: 50
  test_freq: 50
```

---

### 3.3 启动脚本

**创建 SeqKD 启动脚本**：

```bash
#!/bin/bash
set -x

export NCCL_TIMEOUT=36000

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

export WANDB_PROJECT='YOUR_PROJECT_NAME'
export WANDB_API_KEY='YOUR_WANDB_API_KEY'
export HYDRA_FULL_ERROR=1

# SeqKD Training Configuration
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    data.prompt_key=content \
    data.train_files=/path/to/data_with_teacher.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    trainer.critic_warmup=-1 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/tmp/${EXP_NAME} \
    trainer.total_epochs=4 "${@:1}"
```

**关键配置**：
- ❌ **不需要设置** `use_sft_mode=true`（会在训练器中自动设置）
- ✅ **设置** `trainer.critic_warmup=-1`（不使用 Critic）
- ✅ **设置** `rollout.n=8`（生成 8 个响应用于监控）

---

### 3.4 启动训练

```bash
bash seqkd_script.sh \
  --model /path/to/base/model \
  --exp_name seqkd_exp \
  --nnodes 1
```

---

## 四、训练流程

### 4.1 SeqKD 阶段的完整流程

```
1. 数据加载
   ├─ prompts: [p1, p2, ..., p32]
   └─ teacher_response: [t1, t2, ..., t32]
   
2. VLLM 生成（可选，用于监控）
   └─ 生成 256 个响应 (32×8)
   
3. 数据扩展
   └─ batch.repeat(n=8) → 256 个样本
   
4. 设置 SFT 模式
   └─ batch.meta_info["use_sft_mode"] = True
   
5. Actor 训练
   ├─ 选择 teacher 数据
   ├─ 前向传播（teacher_input_ids）
   ├─ 计算 SFT 损失
   └─ 反向传播
```

---

## 五、监控指标

### 5.1 训练指标

- **`actor/sft_loss`**：SFT 损失（应逐渐下降）
- **`actor/teacher_pg_loss`**：同上（兼容性指标）
- **`actor/lr`**：当前学习率
- **`actor/grad_norm`**：梯度范数

### 5.2 验证指标（可选）

- **`val/rouge-L/mean`**：Rouge-L 分数（如果实现）
- **`val/loss`**：验证集损失

---

## 六、与其他阶段的关系

### 6.1 训练流程

```
Stage 1 (SeqKD)  →  Stage 2 (Warmup)  →  Stage 3 (GAD)
   ↓                    ↓                     ↓
SFT 训练            判别器训练            对抗训练
use_sft_mode=True   use_sft_mode=False    use_sft_mode=False
critic_warmup=-1    critic_warmup=10      critic_warmup=0
```

### 6.2 检查点继承

**SeqKD → Warmup**：
```bash
# 1. SeqKD 训练
bash seqkd_script.sh --exp_name seqkd_exp

# 2. 从 SeqKD 检查点继续 Warmup 训练
bash warmup_script.sh \
  --model /path/to/seqkd/checkpoint/actor \
  --reward_model /path/to/reward_model \
  --exp_name warmup_exp
```

**Warmup → GAD**：
```bash
# 3. 从 Warmup 检查点继续 GAD 训练
bash gad_script.sh \
  --model /path/to/warmup/checkpoint/actor \
  --reward_model /path/to/warmup/checkpoint/critic \
  --exp_name gad_exp \
  trainer.critic_warmup=0
```

---

## 七、需要在 Trainer 中添加的逻辑

### ⚠️ 重要：设置 `use_sft_mode` 标志

**文件**：`verl/trainer/ppo/ray_trainer.py`

**位置**：在调用 `actor_rollout_wg.update_actor(batch)` 之前

**需要添加的代码**：

```python
# 在 ray_trainer.py 的 fit 方法中
def fit(self):
    for epoch in range(total_epochs):
        for batch in dataloader:
            # ... 生成、打分等步骤 ...
            
            # 🔥 关键：设置 SFT 模式标志
            if self.config.trainer.critic_warmup < 0:  # SeqKD 阶段
                batch.meta_info["use_sft_mode"] = True
            else:  # Warmup/GAD 阶段
                batch.meta_info["use_sft_mode"] = False
            
            # 更新 Actor
            if self.global_steps > self.config.trainer.critic_warmup:
                actor_output = self.actor_rollout_wg.update_actor(batch)
```

**或者更简单的方式**：

```python
# 在配置中直接设置
if self.config.trainer.critic_warmup < 0:
    # SeqKD mode
    for batch in dataloader:
        batch.meta_info["use_sft_mode"] = True
        actor_output = self.actor_rollout_wg.update_actor(batch)
```

---

## 八、测试验证

### 8.1 单元测试

```python
# 测试 compute_sft_loss
import torch
from verl.trainer.ppo.core_algos import compute_sft_loss

def test_compute_sft_loss():
    log_prob = torch.randn(4, 10)  # (batch, seq_len)
    response_mask = torch.ones(4, 10)
    
    loss = compute_sft_loss(log_prob, response_mask)
    
    assert loss.dim() == 0  # 标量
    print(f"SFT Loss: {loss.item()}")
```

### 8.2 集成测试

```bash
# 1. 运行小规模 SeqKD 训练
bash seqkd_script.sh \
  --model /path/to/model \
  --exp_name seqkd_test \
  trainer.total_epochs=1

# 2. 检查日志
# - 应该看到 actor/sft_loss
# - 应该看到 actor/teacher_pg_loss
# - 不应该看到 critic 相关指标

# 3. 检查检查点
ls /tmp/seqkd_test/global_step_50/actor/
```

---

## 九、常见问题

### Q1: 如何判断是否使用了 SFT 模式？

**A**: 查看训练日志：
- SFT 模式：会输出 `actor/sft_loss` 和 `actor/teacher_pg_loss`
- PPO/GSPO 模式：会输出 `actor/pg_loss` 和 `actor/pg_clipfrac`

### Q2: SeqKD 阶段需要 Critic 吗？

**A**: 不需要。设置 `trainer.critic_warmup=-1` 即可跳过 Critic 更新。

### Q3: 生成的 8 个响应用来做什么？

**A**: 
- **训练时**：不使用，只用 teacher_response
- **验证时**：用于计算 Rouge-L 分数（监控质量）

### Q4: 如何从 SeqKD 切换到 Warmup？

**A**: 
```bash
# SeqKD
trainer.critic_warmup=-1  # 不使用 Critic

# Warmup
trainer.critic_warmup=10  # 前 10 步只训练 Critic
```

---

## 十、总结

### 10.1 修改完成度

| 功能 | 状态 | 说明 |
|------|------|------|
| `compute_sft_loss` 函数 | ✅ 完成 | `core_algos.py` |
| Actor SFT 模式支持 | ✅ 完成 | `dp_actor.py` |
| Teacher 数据加载 | ✅ 完成 | `rl_dataset.py`（之前已完成） |
| 配置参数支持 | ✅ 完成 | 通过 `meta_info` 传递 |
| Trainer 逻辑 | ⚠️ 需要添加 | 设置 `use_sft_mode` 标志 |

### 10.2 核心优势

1. **向后兼容**：不影响 Warmup/GAD 阶段
2. **代码简洁**：复用现有逻辑，无冗余
3. **易于切换**：通过配置控制训练模式
4. **功能完整**：支持完整的 SeqKD 训练流程

### 10.3 下一步

1. ✅ **立即可用**：核心代码已完成
2. ⚠️ **需要添加**：在 `ray_trainer.py` 中设置 `use_sft_mode` 标志
3. 🟢 **可选优化**：添加 Rouge-L 评估（如果需要）

---

## 十一、完整的三阶段训练流程

### 阶段 1：SeqKD（SFT 基线）

```bash
bash seqkd_script.sh \
  --model /path/to/base/model \
  --exp_name seqkd_exp \
  --nnodes 1
```

**配置**：
- `trainer.critic_warmup=-1`
- `use_sft_mode=True`（自动设置）

**输出**：
- `/tmp/seqkd_exp/global_step_XXX/actor/`

---

### 阶段 2：Warmup（判别器训练）

```bash
bash warmup_script.sh \
  --model /tmp/seqkd_exp/global_step_200/actor \
  --reward_model /path/to/reward_model \
  --exp_name warmup_exp \
  --nnodes 1
```

**配置**：
- `trainer.critic_warmup=10`
- `use_sft_mode=False`（默认）

**输出**：
- `/tmp/warmup_exp/global_step_XXX/actor/`
- `/tmp/warmup_exp/global_step_XXX/critic/`

---

### 阶段 3：GAD（对抗训练）

```bash
bash gad_script.sh \
  --model /tmp/warmup_exp/global_step_800/actor \
  --reward_model /tmp/warmup_exp/global_step_800/critic \
  --exp_name gad_exp \
  --nnodes 1 \
  trainer.critic_warmup=0
```

**配置**：
- `trainer.critic_warmup=0`
- `use_sft_mode=False`（默认）

**输出**：
- `/tmp/gad_exp/global_step_XXX/actor/`
- `/tmp/gad_exp/global_step_XXX/critic/`

---

**最终结论**：✅ **Stage 1 (SeqKD) 的核心代码已完成！只需在 Trainer 中添加 `use_sft_mode` 标志设置即可使用。**
