# GAD 代码适配到新 verl 框架的差异分析

## 一、核心差异总览

根据对比分析，旧的 GAD 代码（`gad` 项目）与新的 verl 框架（`gad_gspo` 项目）在 Critic 实现上存在**重大差异**。新框架目前**不支持 GAD 的判别器训练模式**，需要进行大量适配。

---

## 二、主要差异点详解

### 2.1 `_forward_micro_batch` 方法签名

#### 旧代码（gad）
```python
def _forward_micro_batch(self, micro_batch, compute_teacher):
    """
    参数:
        compute_teacher: bool - 是否计算教师回复的值
    """
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
```

#### 新代码（gad_gspo）
```python
def _forward_micro_batch(self, micro_batch):
    """
    参数: 无 compute_teacher 参数
    """
    response_length = micro_batch["responses"].size(-1)
    # 只处理学生回复，没有教师回复的分支
```

**差异说明**：
- 旧代码支持双路前向推理（学生 + 教师）
- 新代码只支持单路推理（学生）
- 需要添加 `compute_teacher` 参数支持

---

### 2.2 值的提取方式（关键差异）

#### 旧代码（gad）- 序列级奖励模型
```python
# 只保留最后一个有效 token 的值
values = values[:, -response_length:]  # 当前 token
response_mask = attention_mask[:, -response_length:]
response_lengths = response_mask.sum(dim=1).long()
last_token_indices = response_lengths - 1

# 创建 mask：只有最后一个 token 为 True
last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
last_token_mask[batch_indices, last_token_indices] = True
values = values * last_token_mask.type_as(values)
```

#### 新代码（gad_gspo）- Token 级 Value 函数
```python
# 保留所有 token 的值（预测下一个 token）
values = values[:, -response_length - 1 : -1].squeeze(-1)
# 没有 last_token_mask 逻辑
```

**差异说明**：
- **旧代码**：判别器被改造为**序列级奖励模型**，整个回复的质量用最后一个 token 的单个标量表示
- **新代码**：标准的 token 级 value 函数，每个 token 都有一个值
- 这是 GAD 的核心设计，必须保留

---

### 2.3 `compute_values` 方法

#### 旧代码（gad）
```python
def compute_values(self, data: DataProto) -> torch.Tensor:
    compute_teacher = data.meta_info["compute_teacher"]  # 🔥 关键参数
    
    if compute_teacher:
        select_keys = ["teacher_response", "teacher_input_ids", 
                      "teacher_attention_mask", "teacher_position_ids"]
        # 特殊处理：teacher forcing for GRPO
        if compute_teacher:
            teacher_repeat = data.meta_info["teacher_repeat"]
            return self._forward_batch_teacher_forcing_grpo(batch, teacher_repeat)
    else:
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    
    # 根据 compute_teacher 调用不同的前向推理
    values = self._forward_micro_batch(micro_batch, compute_teacher=compute_teacher)
```

#### 新代码（gad_gspo）
```python
def compute_values(self, data: DataProto) -> torch.Tensor:
    # 没有 compute_teacher 参数
    select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    # 只计算学生回复的值
    values = self._forward_micro_batch(model_inputs)
```

**差异说明**：
- 旧代码支持通过 `compute_teacher` 元信息控制计算学生还是教师的值
- 新代码完全不支持教师回复的处理
- 旧代码有 `_forward_batch_teacher_forcing_grpo` 方法用于 GRPO 的教师强制

---

### 2.4 `update_critic` 方法（最大差异）

#### 旧代码（gad）- 判别器损失
```python
def update_critic(self, data: DataProto):
    select_keys = [
        "input_ids", "responses", "attention_mask", "position_ids",
        "teacher_input_ids", "teacher_response",  # 🔥 需要教师数据
        "teacher_attention_mask", "teacher_position_ids"
    ]
    
    for epoch in range(self.config.ppo_epochs):
        for mini_batch in dataloader:
            for micro_batch in micro_batches:
                # 双路前向推理
                student_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=False)
                teacher_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=True)
                
                # 计算判别准确率
                d_acc = (teacher_vpreds.sum(dim=-1) > student_vpreds.sum(dim=-1)).float().mean()
                
                # 🔥 使用判别器损失，而非 value loss
                d_loss = core_algos.compute_discriminator_loss(
                    student_vpreds=student_vpreds,
                    teacher_vpreds=teacher_vpreds,
                    response_mask=response_mask,
                    teacher_response_mask=teacher_response_mask,
                )
                
                loss.backward()
                
                metrics = {
                    "critic/d_loss": d_loss.item(),
                    "critic/d_acc": d_acc,
                    "critic/student_value_mean": ...,
                    "critic/teacher_value_mean": ...,
                }
```

#### 新代码（gad_gspo）- 标准 Value Loss
```python
def update_critic(self, data: DataProto):
    select_keys = ["input_ids", "responses", "response_mask", 
                   "attention_mask", "position_ids", "values", "returns"]
    # 没有教师数据
    
    for epoch in range(self.config.ppo_epochs):
        for mini_batch in mini_batches:
            for micro_batch in micro_batches:
                # 只有学生的前向推理
                vpreds = self._forward_micro_batch(model_inputs)
                
                # 🔥 使用标准的 value loss
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=vpreds,
                    values=values,
                    returns=returns,
                    response_mask=response_mask,
                    cliprange_value=self.config.cliprange_value,
                    loss_agg_mode=self.config.loss_agg_mode,
                )
                
                loss.backward()
                
                metrics = {
                    "critic/vf_loss": vf_loss.item(),
                    "critic/vf_clipfrac": vf_clipfrac.item(),
                    "critic/vpred_mean": ...,
                }
```

**差异说明**：
- **旧代码**：使用判别器损失 `compute_discriminator_loss`，对比教师和学生回复
- **新代码**：使用标准 PPO 的 value loss，只优化值函数预测
- 这是 GAD 与标准 PPO 的根本区别

---

### 2.5 判别器损失函数

#### 旧代码（gad）- 存在于 core_algos.py
```python
def compute_discriminator_loss(
    student_vpreds: torch.Tensor, 
    teacher_vpreds: torch.Tensor, 
    response_mask: torch.Tensor, 
    teacher_response_mask: torch.Tensor
):
    """
    判别器损失：让教师得分高于学生得分
    """
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    d_loss = -nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

#### 新代码（gad_gspo）
```python
# ❌ 不存在此函数
```

**差异说明**：
- 新框架的 `core_algos.py` 中完全没有 `compute_discriminator_loss` 函数
- 需要将此函数添加到新框架中

---

### 2.6 特殊方法：Teacher Forcing for GRPO

#### 旧代码（gad）
```python
def _forward_batch_teacher_forcing_grpo(self, batch, teacher_repeat):
    """
    为 GRPO 提供教师强制的值
    teacher_repeat: 每个 prompt 的教师回复重复次数
    """
    response_length = batch["teacher_response"].size(-1)
    input_ids = batch["teacher_input_ids"]
    bsz, seqlen = input_ids.shape
    
    values = torch.zeros((bsz, response_length), device=input_ids.device)
    response_mask = attention_mask[:, -response_length:]
    response_lengths = response_mask.sum(dim=1).long()
    last_token_indices = response_lengths - 1
    
    # 为同一组的教师回复分配递增的值
    for i in range(0, bsz, teacher_repeat):
        for j in range(teacher_repeat):
            values[i + j, last_token_indices[i + j]] = float(j)
    
    return values
```

#### 新代码（gad_gspo）
```python
# ❌ 不存在此方法
```

**差异说明**：
- 这是 GRPO 优势计算的特殊处理
- 为同一 prompt 的多个教师回复分配不同的值，用于相对比较

---

### 2.7 数据字段要求

#### 旧代码（gad）需要的字段
```python
# 学生回复相关
"input_ids", "responses", "attention_mask", "position_ids"

# 教师回复相关（必需）
"teacher_input_ids", "teacher_response", 
"teacher_attention_mask", "teacher_position_ids"

# 元信息
data.meta_info["compute_teacher"]  # 控制计算学生还是教师
data.meta_info["teacher_repeat"]   # GRPO 的教师重复次数
```

#### 新代码（gad_gspo）需要的字段
```python
# 只有学生回复相关
"input_ids", "responses", "attention_mask", "position_ids", 
"values", "returns"

# 没有教师回复相关字段
```

---

## 三、需要修改的文件清单

### 3.1 核心文件

#### 1. `verl/workers/critic/dp_critic.py`
**必须修改**，这是最核心的文件。

**需要的修改**：
- [ ] 修改 `_forward_micro_batch` 方法，添加 `compute_teacher` 参数
- [ ] 添加最后 token 的 mask 逻辑（序列级奖励模型）
- [ ] 修改 `compute_values` 方法，支持 `compute_teacher` 元信息
- [ ] 添加 `_forward_batch_teacher_forcing_grpo` 方法
- [ ] 完全重写 `update_critic` 方法：
  - 添加教师数据的 select_keys
  - 实现双路前向推理
  - 使用判别器损失替代 value loss
  - 添加判别准确率计算
  - 修改 metrics 输出

#### 2. `verl/trainer/ppo/core_algos.py`
**必须添加**判别器损失函数。

**需要的修改**：
- [ ] 添加 `compute_discriminator_loss` 函数
- [ ] 确保函数签名与旧代码一致

#### 3. `verl/utils/dataset/rl_dataset.py`
**必须修改**，支持教师回复的加载。

**需要的修改**：
- [ ] 在 `__getitem__` 方法中添加 `teacher_response` 的处理
- [ ] 对教师回复进行 tokenize 和 postprocess
- [ ] 构建 `teacher_input_ids`（prompt + teacher_response）
- [ ] 生成 `teacher_attention_mask` 和 `teacher_position_ids`

---

### 3.2 可能需要修改的文件

#### 4. `verl/trainer/ppo/ray_trainer.py`
**可能需要修改**，取决于数据流的处理。

**需要检查**：
- [ ] 是否需要在训练循环中设置 `compute_teacher` 元信息
- [ ] 判别器更新的调用是否需要修改
- [ ] 数据批次的准备是否需要包含教师数据

#### 5. `verl/utils/seqlen_balancing.py`
**可能需要修改**，如果使用动态批次大小。

**需要检查**：
- [ ] `rearrange_micro_batches` 是否支持教师数据
- [ ] 动态批次划分时是否正确处理教师回复

---

## 四、详细修改建议

### 4.1 修改 `dp_critic.py` 的 `_forward_micro_batch`

```python
def _forward_micro_batch(self, micro_batch, compute_teacher=False):
    """
    添加 compute_teacher 参数
    """
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
    
    # ... 原有的前向推理代码 ...
    
    # 🔥 关键修改：添加最后 token mask 逻辑
    if self.use_remove_padding:
        values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
        values = values[:, -response_length:]  # 改为当前 token，而非下一个 token
    else:
        values = output.logits
        values = values[:, -response_length:].squeeze(-1)  # 改为当前 token
    
    # 🔥 新增：只保留最后一个有效 token 的值
    response_mask = attention_mask[:, -response_length:]
    response_lengths = response_mask.sum(dim=1).long()
    last_token_indices = response_lengths - 1
    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
    last_token_mask[batch_indices, last_token_indices] = True
    values = values * last_token_mask.type_as(values)
    
    return values
```

### 4.2 添加 `compute_discriminator_loss` 到 `core_algos.py`

```python
def compute_discriminator_loss(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
):
    """
    计算判别器损失，让教师得分高于学生得分
    
    Args:
        student_vpreds: 学生回复的值预测，shape (batch_size, response_length)
        teacher_vpreds: 教师回复的值预测，shape (batch_size, response_length)
        response_mask: 学生回复的 mask
        teacher_response_mask: 教师回复的 mask
    
    Returns:
        d_loss: 判别器损失
    """
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    d_loss = -nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

### 4.3 修改 `update_critic` 方法

```python
def update_critic(self, data: DataProto):
    self.critic_module.train()
    metrics = {}
    
    # 🔥 修改：添加教师数据字段
    select_keys = [
        "input_ids", "responses", "attention_mask", "position_ids",
        "teacher_input_ids", "teacher_response", 
        "teacher_attention_mask", "teacher_position_ids"
    ]
    
    # ... 数据加载代码 ...
    
    for epoch in range(self.config.ppo_epochs):
        for mini_batch in dataloader:
            for micro_batch in micro_batches:
                # 🔥 修改：双路前向推理
                student_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=False)
                teacher_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=True)
                
                # 🔥 新增：计算判别准确率
                d_acc = (teacher_vpreds.sum(dim=-1) > student_vpreds.sum(dim=-1)).float().mean()
                
                # 🔥 修改：使用判别器损失
                d_loss = core_algos.compute_discriminator_loss(
                    student_vpreds=student_vpreds,
                    teacher_vpreds=teacher_vpreds,
                    response_mask=response_mask,
                    teacher_response_mask=teacher_response_mask,
                )
                
                loss = d_loss / self.gradient_accumulation
                loss.backward()
                
                # 🔥 修改：更新 metrics
                metrics.update({
                    "critic/d_loss": d_loss.item(),
                    "critic/d_acc": d_acc.item(),
                    "critic/student_value_mean": masked_sum(student_vpreds, response_mask, axis=-1).mean().item(),
                    "critic/teacher_value_mean": masked_sum(teacher_vpreds, teacher_response_mask, axis=-1).mean().item(),
                })
```

---

## 五、数据流变化

### 5.1 旧代码的数据流

```
数据加载 (rl_dataset.py)
  ↓
提取 teacher_response
  ↓
Tokenize teacher_response
  ↓
构建 teacher_input_ids = prompt + teacher_response
  ↓
生成 teacher_attention_mask, teacher_position_ids
  ↓
传递给 Trainer
  ↓
Trainer 调用 critic.update_critic(data)
  ↓
data 包含学生和教师的所有字段
  ↓
Critic 进行双路前向推理
  ↓
计算判别器损失
```

### 5.2 新代码的数据流（需要适配）

```
数据加载 (rl_dataset.py)
  ↓
❌ 没有 teacher_response 处理
  ↓
只有学生数据
  ↓
传递给 Trainer
  ↓
Trainer 调用 critic.update_critic(data)
  ↓
data 只包含学生字段
  ↓
Critic 只进行单路前向推理
  ↓
计算 value loss（不是判别器损失）
```

---

## 六、关键设计理念差异

### 6.1 Critic 的角色

| 维度 | 旧代码（GAD） | 新代码（标准 PPO） |
|------|--------------|------------------|
| **Critic 类型** | 判别器（Discriminator） | 值函数（Value Function） |
| **输出** | 序列级奖励（单个标量） | Token 级值（每个 token 一个值） |
| **训练目标** | 区分教师和学生回复 | 预测回报 |
| **损失函数** | 判别器损失 | Value Loss |
| **是否需要教师数据** | ✅ 必需 | ❌ 不需要 |

### 6.2 训练流程差异

| 阶段 | 旧代码（GAD） | 新代码（标准 PPO） |
|------|--------------|------------------|
| **数据准备** | Prompt + Teacher Response | Prompt only |
| **生成** | Actor 生成学生回复 | Actor 生成回复 |
| **评分** | 判别器对学生和教师打分 | Reward Model 打分 |
| **优势计算** | GRPO（组内相对比较） | GAE 或其他 |
| **Critic 更新** | 判别器损失（对抗训练） | Value Loss |
| **Actor 更新** | PPO Loss（使用优势） | PPO Loss（使用优势） |

---

## 七、优先级建议

### 高优先级（必须修改）
1. ✅ `dp_critic.py` - 核心文件，必须完全适配
2. ✅ `core_algos.py` - 添加判别器损失函数
3. ✅ `rl_dataset.py` - 支持教师回复加载

### 中优先级（可能需要修改）
4. ⚠️ `ray_trainer.py` - 检查数据流和元信息设置
5. ⚠️ `seqlen_balancing.py` - 检查动态批次处理

### 低优先级（根据需要）
6. 📝 配置文件 - 确保包含 GAD 相关参数
7. 📝 启动脚本 - 确保数据路径和参数正确

---

## 八、验证清单

完成修改后，需要验证以下内容：

- [ ] `_forward_micro_batch` 支持 `compute_teacher` 参数
- [ ] 最后 token mask 逻辑正确实现
- [ ] `compute_discriminator_loss` 函数可用
- [ ] `update_critic` 使用判别器损失
- [ ] 数据加载包含教师回复
- [ ] `teacher_input_ids` 正确构建（prompt + teacher_response）
- [ ] 训练时 metrics 包含 `d_loss` 和 `d_acc`
- [ ] 判别准确率逐渐上升（从 0.5 到 > 0.7）
- [ ] 教师得分高于学生得分

---

## 九、总结

新的 verl 框架（`gad_gspo`）是一个**标准的 PPO 实现**，不支持 GAD 的判别器训练模式。要将旧的 GAD 代码适配到新框架，需要：

1. **核心修改**：将 Critic 从值函数改造为判别器
2. **数据支持**：添加教师回复的加载和处理
3. **损失函数**：添加判别器损失函数
4. **训练流程**：实现双路前向推理和对抗训练

这是一个**非平凡的适配工作**，涉及多个文件的修改，建议按照优先级逐步进行。
