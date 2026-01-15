# Warmup 阶段完整训练流程解析

## 一、Warmup 阶段概述

### 1.1 核心目标

Warmup (Stage 1) 是 GAD 训练的第一阶段，主要目标：
1. **训练判别器**：学习区分教师回复和学生回复的能力
2. **预热学生模型**：让学生模型初步学会生成合理回复
3. **建立基础**：为 Stage 2 的完全对抗训练做准备

### 1.2 关键特性

- **已使用对抗训练**：判别器使用 `compute_discriminator_loss`，而非传统 value loss
- **Critic Warmup 机制**：前 10 步只训练判别器，不更新 Actor
- **数据要求**：必须包含教师回复（`teacher_response`）
- **训练时长**：约 2 epochs，~800 steps

---

## 二、训练配置

### 2.1 启动脚本

```bash
bash scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-warmup-lr1e-6 \
  --nnodes 1
```

### 2.2 关键参数

```python
# 优势估计器
algorithm.adv_estimator=grpo

# 数据配置
data.train_files=/tmp/lmsys_gpt5_chat_4k_filtered_train.parquet
data.train_batch_size=256
data.max_prompt_length=2048
data.max_response_length=1536

# Actor 配置
actor_rollout_ref.model.path=$MODEL_PATH  # 从预训练模型初始化
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.n=8  # 每个prompt生成8个回复
actor_rollout_ref.rollout.temperature=0.8

# Critic (判别器) 配置
critic.model.path=$REWARD_MODEL_PATH  # 从预训练模型初始化
critic.optim.lr=1e-6

# 训练策略
trainer.critic_warmup=10  # 🔥 前10步只训练判别器
trainer.total_epochs=2
trainer.save_freq=50
```

---

## 三、数据准备

### 3.1 数据格式要求

Parquet 文件必须包含以下字段：

```python
{
    "content": [  # prompt (messages格式)
        {"role": "user", "content": "问题内容"}
    ],
    "teacher_response": "教师模型的高质量回复"
}
```

### 3.2 数据加载流程

**位置**: `verl/utils/dataset/rl_dataset.py:213-312`

```python
def __getitem__(self, item):
    row_dict = self.dataframe[item]
    messages = self._build_messages(row_dict)
    
    # 提取教师回复
    teacher_response = None
    if 'teacher_response' in row_dict:
        teacher_response = row_dict.pop('teacher_response')
    
    # 处理 prompt
    raw_prompt = self.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = self.tokenizer(raw_prompt, return_tensors="pt")
    
    # 处理教师回复
    if teacher_response is not None:
        teacher_response = self.tokenizer(
            teacher_response, return_tensors="pt", add_special_tokens=False
        )
        teacher_response_ids, _ = verl_F.postprocess_data(
            input_ids=teacher_response["input_ids"],
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.eos_token_id,
            left_pad=False,  # 回复从右侧填充
            truncation=self.truncation,
        )
        row_dict["teacher_response"] = teacher_response_ids[0]
    
    return row_dict
```

**输出数据**：
- `input_ids`: Prompt 的 token IDs
- `attention_mask`: Prompt 的 attention mask
- `position_ids`: Position IDs
- `teacher_response`: 教师回复的 token IDs

---

## 四、单步训练流程

### 4.1 完整流程图

```
数据加载
  ↓
[Prompt] + [Teacher Response]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
主训练循环 (每个 step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
1. Actor 生成 8 个学生回复
   [Prompt] → Actor → [Student Response 1-8]
  ↓
2. 计算旧策略 log_prob
   [Responses] → Actor → [Old Log Probs]
  ↓
3. 判别器评分（学生回复）
   [Student Responses] → Discriminator → [Scores]
  ↓
4. GRPO 计算优势
   同一 prompt 的 8 个回复相对比较
   → [Advantages: +/- based on group mean]
  ↓
5. 更新判别器（始终执行）
   ┌─────────────────────────────────┐
   │ Forward: Student Response → S_score │
   │ Forward: Teacher Response → T_score │
   │ Loss: -log(sigmoid(T - S))      │
   │ Backward & Update                │
   └─────────────────────────────────┘
  ↓
6. 更新 Actor（step > 10 后执行）
   ┌─────────────────────────────────┐
   │ PPO Loss: clip(ratio * advantage) │
   │ KL Loss: KL(π || π_ref)          │
   │ Backward & Update                │
   └─────────────────────────────────┘
```

### 4.2 代码实现

**位置**: `verl/trainer/ppo/ray_trainer.py:950-1185`

```python
for epoch in range(total_epochs):
    for batch in train_dataloader:
        
        # Step 1: 生成学生回复
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        batch = batch.union(gen_batch_output)
        
        # Step 2: 计算旧策略的 log_prob
        batch.meta_info["compute_teacher"] = False
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        batch = batch.union(old_log_prob)
        
        # Step 3: 使用判别器计算奖励
        batch.meta_info["compute_teacher"] = False
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
        reward_tensor = batch.batch["values"]
        
        # Step 4: 计算优势函数 (GRPO)
        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor
        
        batch = compute_advantage(
            batch,
            adv_estimator='grpo',
            num_repeat=8,
        )
        
        # Step 5: 更新判别器（始终执行）
        critic_output = self.critic_wg.update_critic(batch)
        
        # Step 6: 更新 Actor（warmup 期后）
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_output = self.actor_rollout_wg.update_actor(batch)
```

---

## 五、判别器更新详解

### 5.1 核心认知

**重要**：判别器更新**不依赖** GRPO 优势！

- **GRPO 优势**：用于更新 Actor（学生模型）
- **判别器损失**：用于更新 Critic（判别器）

两者是独立的过程。

### 5.2 判别器更新流程

**位置**: `verl/workers/critic/dp_critic.py:247-337`

```python
def update_critic(self, data: DataProto):
    self.critic_module.train()
    
    # 准备数据：同时需要学生和教师的回复
    select_keys = [
        "input_ids", "responses", "attention_mask", "position_ids",
        "teacher_input_ids", "teacher_response", 
        "teacher_attention_mask", "teacher_position_ids"
    ]
    
    for epoch in range(self.config.ppo_epochs):
        for mini_batch in dataloader:
            for micro_batch in micro_batches:
                
                # 1. 双路前向推理
                student_vpreds = self._forward_micro_batch(
                    micro_batch, compute_teacher=False
                )
                teacher_vpreds = self._forward_micro_batch(
                    micro_batch, compute_teacher=True
                )
                
                # 2. 计算判别准确率（监控）
                d_acc = (teacher_vpreds.sum(dim=-1) > 
                        student_vpreds.sum(dim=-1)).float().mean()
                
                # 3. 计算判别器损失
                d_loss = core_algos.compute_discriminator_loss(
                    student_vpreds=student_vpreds,
                    teacher_vpreds=teacher_vpreds,
                    response_mask=response_mask,
                    teacher_response_mask=teacher_response_mask,
                )
                
                # 4. 反向传播
                loss = d_loss / self.gradient_accumulation
                loss.backward()
                
                # 5. 记录指标
                metrics = {
                    "critic/d_loss": d_loss.item(),
                    "critic/d_acc": d_acc.item(),
                    "critic/student_value_mean": ...,
                    "critic/teacher_value_mean": ...,
                }
            
            # 6. 优化器步进
            self._optimizer_step()
```

### 5.3 判别器损失函数

**位置**: `verl/trainer/ppo/core_algos.py:850-854`

```python
def compute_discriminator_loss(student_vpreds, teacher_vpreds, 
                               response_mask, teacher_response_mask):
    # 计算总分数
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    
    # 对抗损失
    d_loss = -nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

**数学原理**：

```
sigmoid(x) = 1 / (1 + exp(-x))

d_loss = -log(sigmoid(teacher_reward - student_reward))
       = log(1 + exp(student_reward - teacher_reward))
```

**优化目标**：
- 最小化 `d_loss`
- 等价于最大化 `teacher_reward - student_reward`
- 让教师得分尽可能高于学生得分

**梯度方向**：
- 对教师回复：梯度为负，鼓励给**更高分数**
- 对学生回复：梯度为正，鼓励给**更低分数**

### 5.4 判别器前向推理

**位置**: `verl/workers/critic/dp_critic.py:58-149`

```python
def _forward_micro_batch(self, micro_batch, compute_teacher):
    # 根据 compute_teacher 选择输入
    if compute_teacher:
        input_ids = micro_batch["teacher_input_ids"]
        attention_mask = micro_batch["teacher_attention_mask"]
        response_length = micro_batch["teacher_response"].size(-1)
    else:
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        response_length = micro_batch["responses"].size(-1)
    
    # 前向推理
    with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
        output = self.critic_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        values = output.logits
        
        # 关键：只保留最后一个有效 token 的值
        values = values[:, -response_length:]
        response_mask = attention_mask[:, -response_length:]
        response_lengths = response_mask.sum(dim=1).long()
        last_token_indices = response_lengths - 1
        
        # 创建 mask：只有最后一个 token 为 True
        last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        batch_indices = torch.arange(response_mask.size(0))
        last_token_mask[batch_indices, last_token_indices] = True
        
        # 只保留最后 token 的值
        values = values * last_token_mask.type_as(values)
    
    return values
```

**为什么只保留最后 token？**
- 判别器被改造为**序列级奖励模型**
- 整个回复的质量用一个标量表示
- 这个标量放在最后一个有效 token 的位置

---

## 六、GRPO 优势计算（仅用于 Actor）

### 6.1 GRPO 优势的作用

GRPO 优势**仅用于 Actor 更新**，不用于判别器更新。

### 6.2 计算流程

**位置**: `verl/trainer/ppo/core_algos.py:202-263`

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, 
                                   index, norm_adv_by_std_in_grpo=True):
    # 1. 获取每个回复的总分数（来自判别器）
    scores = token_level_rewards.sum(dim=-1)
    
    # 2. 按 uid 分组（同一 prompt 的 8 个回复）
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    # 3. 计算组内统计量
    id2mean = {}
    id2std = {}
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    
    # 4. 标准化优势（相对于组内平均）
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    
    # 5. 广播到 token 级别
    advantages = scores.unsqueeze(-1) * response_mask
    return advantages, advantages
```

### 6.3 GRPO 示例

```
假设一个 prompt 生成了 8 个回复，判别器给分：
scores = [0.5, 0.6, 0.3, 0.7, 0.4, 0.8, 0.5, 0.6]

组内均值: mean = 0.55
组内标准差: std = 0.15

GRPO 优势:
advantages[0] = (0.5 - 0.55) / 0.15 = -0.33  (低于平均，负优势)
advantages[1] = (0.6 - 0.55) / 0.15 = +0.33  (高于平均，正优势)
advantages[3] = (0.7 - 0.55) / 0.15 = +1.00  (远高于平均)
advantages[5] = (0.8 - 0.55) / 0.15 = +1.67  (最高分，最大正优势)
```

**GRPO 的作用**：
- 对同一 prompt 的多个回复进行**相对比较**
- 高于平均的回复获得正优势（鼓励）
- 低于平均的回复获得负优势（惩罚）
- 减少方差，提高训练稳定性

---

## 七、Actor 更新（仅在 Warmup 期后）

### 7.1 Warmup 机制

```python
# 只有在 warmup 期过后才更新 Actor
if self.config.trainer.critic_warmup <= self.global_steps:
    actor_output = self.actor_rollout_wg.update_actor(batch)
```

**Warmup 期时间线**：
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
└─ ✅ 更新 Actor
```

### 7.2 Actor 更新使用的数据

```python
{
    "responses": ...,           # 学生回复
    "old_log_probs": ...,       # 旧策略的 log 概率
    "advantages": ...,          # GRPO 优势
    "returns": ...,             # 回报（在 GRPO 中等于 advantages）
}
```

### 7.3 PPO 损失函数

```python
def compute_policy_loss(log_probs, old_log_probs, advantages, cliprange):
    # 计算重要性采样比率
    ratio = torch.exp(log_probs - old_log_probs)
    
    # PPO clipped loss
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - cliprange, 1.0 + cliprange
    )
    pg_loss = torch.max(pg_losses1, pg_losses2).mean()
    
    return pg_loss
```

---

## 八、两个更新过程对比

### 8.1 判别器更新 vs Actor 更新

| 维度 | 判别器更新 (Critic) | Actor 更新 (Student Model) |
|------|-------------------|---------------------------|
| **输入数据** | Student + Teacher Responses | Student Responses |
| **损失函数** | `-log(sigmoid(T_score - S_score))` | `PPO Loss + KL Loss` |
| **优化目标** | 区分教师和学生 | 增加高优势回复概率 |
| **是否使用优势** | ❌ 不使用 | ✅ 使用 GRPO 优势 |
| **是否使用教师回复** | ✅ 使用 | ❌ 不使用 |
| **更新时机** | 每个 step | Step > 10 后 |

### 8.2 完整信息流图

```
┌──────────────────────────────────────────────────────────┐
│                    数据准备阶段                           │
└──────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
  [Prompt]                          [Teacher Response]
        │                                   │
        └─────────────┬─────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│              Actor 生成学生回复                           │
│  Prompt → Actor → [Student Response 1-8]                 │
└──────────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│           判别器评分（为 GRPO 提供原始分数）               │
│  Student Responses → Discriminator → [Scores]            │
└──────────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│              GRPO 计算相对优势                            │
│  Scores → Group Normalize → [Advantages]                 │
└──────────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────────┐
        ↓                               ↓
┌─────────────────┐          ┌─────────────────┐
│  更新判别器      │          │   更新 Actor     │
│  (始终执行)      │          │  (Step > 10)     │
└─────────────────┘          └─────────────────┘
        │                               │
        ↓                               ↓
    使用判别器损失                   使用 PPO 损失
   (不依赖优势)                     (依赖优势)
```

---

## 九、监控指标

### 9.1 判别器指标

- **`critic/d_loss`**: 判别器损失
  - 初期较大（判别能力弱）
  - 逐渐下降（学会区分）
  - 理想值：0.3-0.5

- **`critic/d_acc`**: 判别准确率
  - 教师得分 > 学生得分的比例
  - 初期接近 0.5（随机猜测）
  - 逐渐上升（学会判别）
  - 理想值：> 0.7

- **`critic/student_value_mean`**: 学生回复平均分
  - 初期较低
  - 随训练逐渐上升

- **`critic/teacher_value_mean`**: 教师回复平均分
  - 应保持稳定且较高
  - 作为学生的目标

### 9.2 Actor 指标（Step > 10 后）

- **`actor/loss`**: PPO 策略损失
- **`actor/kl`**: 与参考策略的 KL 散度
- **`actor/entropy`**: 策略熵
- **`actor/clipfrac`**: 被裁剪的比例

### 9.3 数据指标

- **`data/response_length`**: 生成回复的平均长度
- **`training/global_token_num`**: 有效 token 数量

---

## 十、Warmup 阶段的训练目标

### 10.1 短期目标（前 10 步）

1. 判别器学会区分教师和学生回复
2. `d_acc` 从 0.5 上升到 > 0.6
3. 建立稳定的奖励信号

### 10.2 中期目标（10-800 步）

1. 判别器持续改进判别能力
2. Actor 根据判别器反馈优化策略
3. 学生回复质量逐渐提升

### 10.3 长期目标（整个 Warmup）

1. `d_acc` 达到 0.7-0.8
2. 学生回复分数接近教师（但仍有差距）
3. 为 GAD Stage 2 的完全对抗训练做好准备

---

## 十一、关键设计细节

### 11.1 为什么判别器不使用 GRPO 优势？

**判别器的训练目标**：
- 学习一个评分函数：`f(response) → score`
- 使得：`f(teacher_response) > f(student_response)`
- 这是一个**二分类问题**

**GRPO 优势的局限性**：
- 优势是**相对的**（相对于组内平均）
- 不同 prompt 的优势不可比较
- 优势可能为正，但绝对分数仍然很低

**判别器需要学习绝对质量**，而不是相对排名。

### 11.2 为什么需要 Critic Warmup？

**Warmup 期的意义**：
1. 让判别器先建立初步的判别能力
2. 避免训练初期判别器和 Actor 都不稳定
3. 给判别器一个"领先优势"
4. 建立稳定的奖励信号

**课程学习策略**：
- 先易后难
- 判别器先行
- 逐步引入对抗

---

## 十二、总结

### 12.1 Warmup 阶段的本质

1. **已经是对抗训练**，不是传统的 RL
2. 使用判别器损失，而非 value loss
3. 通过 `critic_warmup=10` 让判别器先建立能力
4. 为后续的 GAD Stage 2 打好基础

### 12.2 核心要点

- **判别器更新不依赖 GRPO 优势**
  - 使用判别器损失：`-log(sigmoid(T_score - S_score))`
  - 直接对比教师和学生回复
  - 学习绝对质量评估

- **GRPO 优势仅用于 Actor 更新**
  - 将判别器的绝对分数转换为相对优势
  - 用于 PPO 损失计算
  - 指导学生模型改进

- **两个更新过程独立**
  - 判别器：学习区分能力
  - Actor：学习生成能力
  - 通过判别器的分数连接

### 12.3 与 GAD Stage 2 的区别

| 维度 | Warmup (Stage 1) | GAD (Stage 2) |
|------|------------------|---------------|
| **Critic Warmup** | 10 steps | 0 steps |
| **初始化** | 从预训练模型 | 从 Warmup checkpoint |
| **训练难度** | 简单（判别器有优势） | 困难（完全对抗） |
| **训练时长** | 2 epochs (~800 steps) | 4 epochs |

### 12.4 训练策略

- **课程学习**：先易后难
- **判别器先行**：建立稳定的奖励信号
- **对抗训练**：学生和判别器相互博弈

这是一个精心设计的训练流程，为 GAD 的成功奠定了基础！
