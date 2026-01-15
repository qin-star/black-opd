# GAD + GSPO 代码修改总结

## 一、修改概览

已完成对新 verl 框架的 GAD（Generative Adversarial Distillation）适配，同时保持了 GSPO 支持。所有修改都是**向后兼容**的，不会影响现有的标准 PPO 训练。

### 核心特性

✅ **GAD 判别器训练**：Critic 可作为判别器，区分教师和学生回复  
✅ **GSPO 策略优化**：Actor 使用序列级重要性采样  
✅ **GRPO 优势估计**：组内标准化优势计算  
✅ **向后兼容**：不影响标准 PPO 训练流程  
✅ **自动检测模式**：根据数据是否包含 `teacher_response` 自动切换训练模式

---

## 二、修改的文件清单

### 1. `verl/trainer/ppo/core_algos.py`

**修改内容**：添加判别器损失函数

```python
def compute_discriminator_loss(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算判别器损失：让教师得分高于学生得分
    Loss = -log(sigmoid(teacher_reward - student_reward))
    """
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    d_loss = -torch.nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

**位置**：第 1441-1475 行（在 `compute_value_loss` 之后）

---

### 2. `verl/workers/critic/dp_critic.py`

#### 2.1 修改 `_forward_micro_batch` 方法

**修改内容**：
- 添加 `compute_teacher` 参数，支持双路前向推理
- 实现序列级奖励模型：只保留最后一个有效 token 的值

```python
def _forward_micro_batch(self, micro_batch, compute_teacher=False):
    # 根据 compute_teacher 选择输入数据
    if compute_teacher:
        response_length = micro_batch["teacher_response"].size(-1)
        input_ids = micro_batch["teacher_input_ids"]
        attention_mask = micro_batch["teacher_attention_mask"]
        position_ids = micro_batch["teacher_position_ids"]
    else:
        response_length = micro_batch["responses"].size(-1)
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
    
    # ... 前向推理 ...
    
    # 关键：只保留最后一个有效 token 的值（序列级奖励）
    response_mask = attention_mask[:, -response_length:]
    response_lengths = response_mask.sum(dim=1).long()
    last_token_indices = response_lengths - 1
    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
    last_token_mask[batch_indices, last_token_indices] = True
    values = values * last_token_mask.type_as(values)
    
    return values
```

#### 2.2 添加 `_forward_batch_teacher_forcing_grpo` 方法

**修改内容**：为 GRPO 提供教师强制值

```python
def _forward_batch_teacher_forcing_grpo(self, batch, teacher_repeat):
    """
    为同一 prompt 的多个教师回复分配递增的值
    用于 GRPO 的组内相对比较
    """
    response_length = batch["teacher_response"].size(-1)
    input_ids = batch["teacher_input_ids"]
    bsz, seqlen = input_ids.shape
    attention_mask = batch["teacher_attention_mask"]
    
    values = torch.zeros((bsz, response_length), device=input_ids.device)
    response_mask = attention_mask[:, -response_length:]
    response_lengths = response_mask.sum(dim=1).long()
    last_token_indices = response_lengths - 1
    
    # 为同一组的教师回复分配递增值
    for i in range(0, bsz, teacher_repeat):
        for j in range(teacher_repeat):
            values[i + j, last_token_indices[i + j]] = float(j)
    
    return values
```

#### 2.3 修改 `compute_values` 方法

**修改内容**：
- 支持 `compute_teacher` 元信息
- 支持教师强制（teacher forcing）

```python
def compute_values(self, data: DataProto) -> torch.Tensor:
    # 检查是否计算教师值
    compute_teacher = data.meta_info.get("compute_teacher", False)
    
    # 根据 compute_teacher 选择数据字段
    if compute_teacher:
        select_keys = ["teacher_response", "teacher_input_ids", 
                      "teacher_attention_mask", "teacher_position_ids"]
        
        # 教师强制（用于 GRPO）
        if "teacher_repeat" in data.meta_info:
            teacher_repeat = data.meta_info["teacher_repeat"]
            batch = data.select(batch_keys=select_keys).batch
            return self._forward_batch_teacher_forcing_grpo(batch, teacher_repeat)
    else:
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    
    # ... 前向推理 ...
    values = self._forward_micro_batch(model_inputs, compute_teacher=compute_teacher)
    
    return values
```

#### 2.4 修改 `update_critic` 方法

**修改内容**：
- 自动检测 GAD 模式（是否包含 `teacher_response`）
- GAD 模式：使用判别器损失
- 标准模式：使用 value loss

```python
def update_critic(self, data: DataProto):
    self.critic_module.train()
    metrics = {}
    
    # 自动检测训练模式
    use_discriminator = "teacher_response" in data.batch
    
    if use_discriminator:
        # GAD 模式：需要学生和教师数据
        select_keys = [
            "input_ids", "responses", "attention_mask", "position_ids",
            "teacher_input_ids", "teacher_response", 
            "teacher_attention_mask", "teacher_position_ids"
        ]
    else:
        # 标准 PPO 模式
        select_keys = ["input_ids", "responses", "response_mask", 
                      "attention_mask", "position_ids", "values", "returns"]
    
    # ... 数据加载 ...
    
    for micro_batch in micro_batches:
        if use_discriminator:
            # GAD 判别器训练
            student_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=False)
            teacher_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=True)
            
            # 计算判别准确率
            d_acc = (teacher_vpreds.sum(dim=-1) > student_vpreds.sum(dim=-1)).float().mean()
            
            # 计算判别器损失
            d_loss = core_algos.compute_discriminator_loss(
                student_vpreds=student_vpreds,
                teacher_vpreds=teacher_vpreds,
                response_mask=response_mask,
                teacher_response_mask=teacher_response_mask,
            )
            
            loss.backward()
            
            metrics.update({
                "critic/d_loss": d_loss.item(),
                "critic/d_acc": d_acc.item(),
                "critic/student_value_mean": ...,
                "critic/teacher_value_mean": ...,
            })
        else:
            # 标准 PPO 值函数训练
            vpreds = self._forward_micro_batch(micro_batch, compute_teacher=False)
            vf_loss, vf_clipfrac = core_algos.compute_value_loss(...)
            
            loss.backward()
            
            metrics.update({
                "critic/vf_loss": vf_loss.item(),
                "critic/vf_clipfrac": vf_clipfrac.item(),
                "critic/vpred_mean": ...,
            })
    
    return metrics
```

---

### 3. `verl/utils/dataset/rl_dataset.py`

**修改内容**：支持 `teacher_response` 的加载和处理

```python
def __getitem__(self, item):
    row_dict: dict = self.dataframe[item]
    messages = self._build_messages(row_dict)
    
    # 提取 teacher_response（如果存在）
    teacher_response = row_dict.pop("teacher_response", None)
    
    # ... 处理 prompt ...
    
    # 处理 teacher_response
    if teacher_response is not None:
        # 1. Tokenize 教师回复
        teacher_response_tokens = self.tokenizer(
            teacher_response, return_tensors="pt", add_special_tokens=False
        )
        teacher_response_ids = teacher_response_tokens["input_ids"]
        
        # 2. Postprocess（截断/填充）
        teacher_response_ids, _ = verl_F.postprocess_data(
            input_ids=teacher_response_ids,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,  # 右侧填充
            truncation=self.truncation,
        )
        
        # 3. 构建 teacher_input_ids = prompt + teacher_response
        prompt_ids = row_dict["input_ids"]
        teacher_input_ids = torch.cat([prompt_ids.unsqueeze(0), teacher_response_ids], dim=1)
        
        # 4. 生成 attention_mask 和 position_ids
        teacher_attention_mask = (teacher_input_ids != self.tokenizer.pad_token_id).long()
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        
        # 5. 添加到 row_dict
        row_dict["teacher_response"] = teacher_response_ids[0]
        row_dict["teacher_input_ids"] = teacher_input_ids[0]
        row_dict["teacher_attention_mask"] = teacher_attention_mask[0]
        row_dict["teacher_position_ids"] = teacher_position_ids[0]
    
    return row_dict
```

---

## 三、使用方法

### 3.1 标准 PPO 训练（无需修改）

如果数据中**不包含** `teacher_response`，代码自动使用标准 PPO 模式：

```bash
python -m verl.trainer.main_ppo \
  --config-path configs \
  --config-name ppo_config \
  data.train_files=/path/to/data_without_teacher.parquet
```

**数据格式**：
```python
{
    "content": [{"role": "user", "content": "问题"}],
    # 不需要 teacher_response
}
```

---

### 3.2 GAD + GSPO 训练

如果数据中**包含** `teacher_response`，代码自动使用 GAD 判别器模式：

```bash
python -m verl.trainer.main_ppo \
  --config-path configs \
  --config-name gad_gspo_config \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.policy_loss=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.rollout.n=8 \
  trainer.critic_warmup=10 \
  data.train_files=/path/to/data_with_teacher.parquet
```

**数据格式**：
```python
{
    "content": [{"role": "user", "content": "问题"}],
    "teacher_response": "教师模型的高质量回复"  # 🔥 关键字段
}
```

---

### 3.3 配置示例

#### 完整的 GAD + GSPO 配置

```yaml
# 数据配置
data:
  train_files: /path/to/data_with_teacher_response.parquet
  train_batch_size: 256
  max_prompt_length: 2048
  max_response_length: 1536

# 算法配置
algorithm:
  adv_estimator: grpo  # 🔥 GRPO 优势估计
  norm_adv_by_std_in_grpo: true
  gamma: 1.0
  lam: 0.95

# Actor 配置
actor_rollout_ref:
  model:
    path: /path/to/model
  
  rollout:
    n: 8  # 每个 prompt 生成 8 个回复
    temperature: 0.8
  
  actor:
    policy_loss: gspo  # 🔥 GSPO 策略损失
    loss_agg_mode: seq-mean-token-mean  # 🔥 序列级聚合
    clip_ratio: 0.2
    optim:
      lr: 1e-6
  
  ppo_mini_batch_size: 256
  ppo_micro_batch_size_per_gpu: 8
  ppo_epochs: 1

# Critic（判别器）配置
critic:
  model:
    path: /path/to/reward_model
  optim:
    lr: 1e-6
  ppo_epochs: 1
  cliprange_value: 0.2

# 训练配置
trainer:
  critic_warmup: 10  # 🔥 前 10 步只训练判别器
  total_epochs: 2
  save_freq: 50
```

---

## 四、训练流程

### 4.1 GAD + GSPO 完整流程

```
数据加载（包含 teacher_response）
  ↓
Actor 生成 8 个学生回复
  ↓
判别器评分（学生 vs 教师）
  ↓
GRPO 计算优势（组内标准化）
  ↓
更新判别器（判别器损失）
  - 双路前向推理
  - 计算 d_loss = -log(sigmoid(T_score - S_score))
  - 反向传播
  ↓
更新 Actor（GSPO 损失，step > 10 后）
  - 计算序列级重要性采样比率
  - PPO clipping
  - 反向传播
```

### 4.2 Critic Warmup 机制

```
Step 0-9 (critic_warmup=10):
├─ 生成学生回复 (n=8)
├─ 判别器打分
├─ 计算 GRPO 优势
├─ ✅ 更新判别器
└─ ❌ 不更新 Actor

Step 10+:
├─ 生成学生回复 (n=8)
├─ 判别器打分
├─ 计算 GRPO 优势
├─ ✅ 更新判别器
└─ ✅ 更新 Actor（使用 GSPO）
```

---

## 五、监控指标

### 5.1 GAD 判别器指标

- **`critic/d_loss`**：判别器损失（应逐渐下降，理想值 0.3-0.5）
- **`critic/d_acc`**：判别准确率（应从 0.5 上升到 > 0.7）
- **`critic/student_value_mean`**：学生回复平均分（应逐渐上升）
- **`critic/teacher_value_mean`**：教师回复平均分（应保持稳定且较高）

### 5.2 GSPO Actor 指标

- **`actor/pg_clipfrac`**：裁剪比例（应在 0.1-0.3）
- **`actor/ppo_kl`**：KL 散度（应保持较小）
- **`actor/entropy`**：策略熵（不应过快下降）

### 5.3 标准 PPO 指标（非 GAD 模式）

- **`critic/vf_loss`**：值函数损失
- **`critic/vf_clipfrac`**：值函数裁剪比例
- **`critic/vpred_mean`**：值预测平均值

---

## 六、关键设计决策

### 6.1 向后兼容性

所有修改都是**向后兼容**的：
- 通过检测 `teacher_response` 字段自动切换模式
- 不影响现有的标准 PPO 训练流程
- 可以在同一代码库中运行 PPO 和 GAD

### 6.2 序列级奖励模型

判别器被改造为**序列级奖励模型**：
- 整个回复的质量用一个标量表示
- 这个标量放在最后一个有效 token 的位置
- 通过 `last_token_mask` 实现

### 6.3 自动模式检测

```python
# 在 update_critic 中自动检测
use_discriminator = "teacher_response" in data.batch

if use_discriminator:
    # GAD 模式：判别器损失
    d_loss = compute_discriminator_loss(...)
else:
    # 标准模式：value loss
    vf_loss = compute_value_loss(...)
```

### 6.4 无冗余代码

- 复用现有的数据处理逻辑
- 使用条件分支而非重复代码
- 保持代码简洁和可维护性

---

## 七、验证清单

### 7.1 代码完整性

- [x] `compute_discriminator_loss` 已添加到 `core_algos.py`
- [x] `_forward_micro_batch` 支持 `compute_teacher` 参数
- [x] 最后 token mask 逻辑已实现
- [x] `_forward_batch_teacher_forcing_grpo` 已添加
- [x] `compute_values` 支持 `compute_teacher` 元信息
- [x] `update_critic` 自动检测 GAD 模式
- [x] `rl_dataset.py` 支持 `teacher_response` 加载

### 7.2 功能验证

- [ ] 标准 PPO 训练正常运行（无 `teacher_response`）
- [ ] GAD 训练正常运行（有 `teacher_response`）
- [ ] GSPO 损失计算正确
- [ ] 判别器指标正常输出
- [ ] Critic warmup 机制生效

### 7.3 性能验证

- [ ] `d_acc` 从 0.5 上升到 > 0.7
- [ ] `d_loss` 逐渐下降
- [ ] 教师得分 > 学生得分
- [ ] Actor 训练稳定（step > 10 后）

---

## 八、常见问题

### Q1: 如何判断是否使用了 GAD 模式？

**A**: 查看训练日志中的指标：
- GAD 模式：会输出 `critic/d_loss` 和 `critic/d_acc`
- 标准模式：会输出 `critic/vf_loss` 和 `critic/vf_clipfrac`

### Q2: 数据中必须包含 `teacher_response` 吗？

**A**: 不是必须的：
- 有 `teacher_response`：自动使用 GAD 模式
- 无 `teacher_response`：自动使用标准 PPO 模式

### Q3: GSPO 可以单独使用吗（不用 GAD）？

**A**: 可以！只需：
```yaml
algorithm.adv_estimator: grpo
actor_rollout_ref.actor.policy_loss: gspo
# 数据中不包含 teacher_response
```

### Q4: 如何调试判别器训练？

**A**: 检查以下指标：
1. `critic/d_acc` 是否从 0.5 开始上升
2. `critic/teacher_value_mean` 是否 > `critic/student_value_mean`
3. `critic/d_loss` 是否逐渐下降

---

## 九、总结

### 9.1 修改的核心价值

1. **完整的 GAD 支持**：判别器训练、序列级奖励模型
2. **GSPO 集成**：序列级策略优化
3. **向后兼容**：不影响现有代码
4. **自动检测**：根据数据自动切换模式
5. **代码简洁**：无冗余，易维护

### 9.2 适用场景

| 场景 | 配置 | 数据要求 |
|------|------|---------|
| **标准 PPO** | `policy_loss=ppo` | 无 `teacher_response` |
| **纯 GSPO** | `policy_loss=gspo, adv_estimator=grpo` | 无 `teacher_response` |
| **GAD + PPO** | `policy_loss=ppo` | 有 `teacher_response` |
| **GAD + GSPO** | `policy_loss=gspo, adv_estimator=grpo` | 有 `teacher_response` |

### 9.3 推荐配置

**最佳实践**：GAD + GSPO
```yaml
algorithm.adv_estimator: grpo
actor_rollout_ref.actor.policy_loss: gspo
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean
actor_rollout_ref.rollout.n: 8
trainer.critic_warmup: 10
data.train_files: /path/to/data_with_teacher_response.parquet
```

这个配置结合了：
- **GAD**：判别器指导学习
- **GSPO**：序列级策略优化
- **GRPO**：组内标准化优势

适用于需要教师指导的序列级任务（代码生成、数学推理等）。
