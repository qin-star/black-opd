# GAD (Generative Adversarial Distillation) 完整训练流程与代码设计解析

> 本文档详细解析了GAD训练阶段的完整流程、核心设计原理和代码实现细节，包括Critic判别器的更新机制、Actor生成器的优化过程以及两者之间的对抗训练关系。

## 目录
- [一、GAD训练阶段概述](#一gad训练阶段概述)
- [二、核心设计原理](#二核心设计原理)
- [三、完整训练流程](#三完整训练流程)
- [四、Critic判别器更新机制详解](#四critic判别器更新机制详解)
- [五、Critic分数计算原理](#五critic分数计算原理)
- [六、数据流设计](#六数据流设计)
- [七、训练方法的理论基础](#七训练方法的理论基础)
- [八、关键超参数分析](#八关键超参数分析)
- [九、训练监控指标](#九训练监控指标)
- [十、总结](#十总结)

---

## 一、GAD训练阶段概述

GAD (Generative Adversarial Distillation) 训练阶段是在Warmup阶段之后进行的对抗训练阶段。该阶段的核心思想是：**将VeRL框架中的Critic模块hack成Discriminator（判别器）**，通过对抗训练的方式让学生模型生成的响应逼近教师模型（GPT-5）的响应质量。

### 训练配置特点

从训练脚本 [gpt5-chat-filtered-7b-adversarial-lr1e-6.sh](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh:0:0-0:0) 可以看出：

- **继承Warmup检查点**：从Warmup阶段的第800步检查点继续训练
- **同时加载Actor和Critic**：
  - Actor（生成器）：从warmup的actor检查点加载
  - Critic（判别器）：从warmup的critic检查点加载
- **训练超参数**：
  - Actor学习率：1e-6
  - Critic学习率：1e-6
  - 训练轮数：4 epochs
  - 批次大小：256
  - 采样数：n=8（每个prompt生成8个响应）
  - KL散度系数：0.001
  - 温度：0.8

## 二、核心设计原理

### 2.1 Critic作为Discriminator的Hack实现

**关键代码位置**：[verl/verl/workers/critic/dp_critic.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/critic/dp_critic.py:0:0-0:0)

#### 判别器的前向传播逻辑

```python
def _forward_micro_batch(self, micro_batch, compute_teacher):
    # 根据compute_teacher决定处理学生还是教师响应
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

**核心Hack点**：
1. **值预测改为奖励预测**：原本Critic预测每个token的value，现在改为预测奖励分数
2. **只在最后一个token输出分数**：
```python
# 只保留最后一个有效token的值
values = values[:, -response_length:]
response_mask = attention_mask[:, -response_length:]
response_lengths = response_mask.sum(dim=1).long()
last_token_indices = response_lengths - 1
last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
last_token_mask[batch_indices, last_token_indices] = True
values = values * last_token_mask.type_as(values)
```

### 2.2 判别器损失函数

**关键代码位置**：[verl/verl/trainer/ppo/core_algos.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/trainer/ppo/core_algos.py:0:0-0:0)

```python
def compute_discriminator_loss(student_vpreds, teacher_vpreds, response_mask, teacher_response_mask):
    # 计算教师和学生响应的总分数
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    
    # 使用二元交叉熵损失，期望teacher_reward > student_reward
    d_loss = -nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

**设计原理**：
- 判别器的目标是区分教师响应和学生响应
- 损失函数鼓励 `teacher_reward > student_reward`
- 使用logsigmoid确保数值稳定性
- 这是一个二分类问题：真实样本（教师）vs 生成样本（学生）

### 2.3 生成器（Actor）的优势函数计算

**关键代码位置**：[verl/verl/trainer/ppo/ray_trainer.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/trainer/ppo/ray_trainer.py:0:0-0:0) 中的 [compute_advantage](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/trainer/ppo/ray_trainer.py:200:0-275:15) 函数

GAD使用GRPO (Group Relative Policy Optimization) 作为优势估计器：

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=True):
    # 1. 计算每个响应的总分数
    scores = token_level_rewards.sum(dim=-1)
    
    # 2. 按照prompt分组（同一个prompt的n个响应）
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    # 3. 计算组内均值和标准差
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    
    # 4. 归一化优势
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    
    return scores, scores
```

**设计原理**：
- **组内比较**：每个prompt生成n=8个响应，在这8个响应内部进行相对比较
- **归一化**：减去组内均值，除以组内标准差，使得优势值标准化
- **相对优势**：不依赖绝对奖励值，只关心相对好坏

## 三、完整训练流程

### 3.1 主训练循环（fit函数）

**代码位置**：[verl/verl/trainer/ppo/ray_trainer.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/trainer/ppo/ray_trainer.py:0:0-0:0) 的 [fit](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/trainer/ppo/ray_trainer.py:917:4-1198:26) 方法（918-1199行）

#### Step 1: 生成阶段（Generation）

```python
# 使用Actor生成n=8个响应
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

# 重复batch以对齐n个响应
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**关键点**：
- 每个prompt生成8个不同的响应（temperature=0.8保证多样性）
- 使用VLLM进行高效推理
- 生成的响应会与原始prompt拼接

#### Step 2: 计算旧策略的对数概率（Old Log Prob）

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
entropys = old_log_prob.batch["entropys"]
batch = batch.union(old_log_prob)
```

**作用**：
- 计算当前策略下生成响应的对数概率
- 用于后续PPO的重要性采样比率计算
- 同时计算熵用于熵正则化

#### Step 3: 计算参考策略的对数概率（Reference Log Prob）

```python
if self.use_reference_policy:
    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
```

**作用**：
- 计算参考策略（冻结的初始模型）的对数概率
- 用于KL散度惩罚，防止策略偏离太远

#### Step 4: 判别器打分（核心创新点）

```python
# 使用Critic（判别器）对学生响应打分
values = self.critic_wg.compute_values(batch)
batch = batch.union(values)
reward_tensor = batch.batch["values"]  # (bsz, response_length)
```

**关键实现**：
- [compute_values](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/critic/dp_critic.py:168:4-217:21) 函数会调用 [_forward_micro_batch(compute_teacher=False)](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/actor/dp_actor.py:81:4-269:37)
- 判别器输出的是学生响应的分数
- 这个分数会作为token_level_scores

#### Step 5: 应用KL惩罚

```python
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
    metrics.update(kl_metrics)
```

**公式**：
```
token_level_rewards = token_level_scores - beta * KL(π_current || π_ref)
```

**作用**：
- 防止策略更新过快
- 保持生成的多样性
- beta=0.001（KL系数）

#### Step 6: 计算优势函数

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,  # GRPO
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,  # 8
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
)
```

**GRPO优势计算流程**：
1. 将8个响应按照uid分组
2. 计算组内均值和标准差
3. 归一化：`advantage = (score - mean) / std`
4. 分数高的响应获得正优势，分数低的获得负优势

#### Step 7: 更新判别器（Critic）

```python
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
```

**更新逻辑**（[dp_critic.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/critic/dp_critic.py:0:0-0:0) 的 [update_critic](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/critic/dp_critic.py:219:4-310:22) 方法）：

```python
# 前向传播计算学生和教师的分数
student_vpreds = self._forward_micro_batch(data, compute_teacher=False)
teacher_vpreds = self._forward_micro_batch(data, compute_teacher=True)

# 计算判别准确率
d_acc = (teacher_vpreds.sum(dim=-1) > student_vpreds.sum(dim=-1)).float().mean()

# 计算判别器损失
d_loss = core_algos.compute_discriminator_loss(
    student_vpreds=student_vpreds,
    teacher_vpreds=teacher_vpreds,
    response_mask=response_mask,
    teacher_response_mask=teacher_response_mask,
)

# 反向传播和优化
loss.backward()
self._optimizer_step()
```

**训练细节**：
- PPO epochs: 1
- Mini-batch size: 256
- Gradient clipping: 0.2
- 使用FSDP进行分布式训练

#### Step 8: 更新生成器（Actor）

```python
if self.config.trainer.critic_warmup <= self.global_steps:
    actor_output = self.actor_rollout_wg.update_actor(batch)
```

**更新逻辑**（[dp_actor.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/actor/dp_actor.py:0:0-0:0) 的 [update_policy](cci:1://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/workers/actor/dp_actor.py:374:4-525:22) 方法）：

```python
# 前向传播计算新的log_prob
entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

# 计算PPO策略损失
pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,  # 来自判别器的分数
    response_mask=response_mask,
    cliprange=clip_ratio,  # 0.2
)

# 熵正则化
entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask)
policy_loss = pg_loss - entropy_loss * entropy_coeff

# KL损失（额外的正则化）
if self.config.use_kl_loss:
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask)
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef  # 0.001

# 反向传播
loss.backward()
self._optimizer_step()
```

**PPO策略损失公式**：
```
L_CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
```
其中：
- `r(θ) = π_θ(a|s) / π_old(a|s)` 是重要性采样比率
- `A` 是优势函数（来自判别器）
- `ε = 0.2` 是裁剪范围

**总损失**：
```
Loss = L_CLIP - entropy_coeff * H(π) + kl_loss_coef * KL(π || π_ref)
```

---

## 六、数据流设计

### 6.1 数据加载

**代码位置**：[verl/verl/utils/dataset/rl_dataset.py](cci:7://file:///d:/%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6/RAG%E5%BC%80%E5%8F%91/Query_RL/Program/OPD/LMOps/gad/verl/verl/utils/dataset/rl_dataset.py:0:0-0:0)

数据格式要求：
```python
{
    "content": [...],  # prompt的对话历史
    "teacher_response": "教师模型的响应文本"
}
```

**处理流程**：
1. 从parquet文件加载数据
2. 提取 `teacher_response` 字段
3. 分别tokenize prompt和teacher_response
4. 对teacher_response进行右填充（因为是响应）
5. 对prompt进行左填充（标准做法）

### 6.2 Rollout阶段的数据扩充

**代码位置**：`verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

```python
# 学生响应
seq = torch.cat([idx, response], dim=-1)
response_position_ids = position_ids[..., -1:] + delta_position_id
position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

# 教师响应
teacher_seq = torch.cat([idx, teacher_response], dim=-1)
teacher_response_position_ids = position_ids[..., -1:] + teacher_delta_position_id
teacher_position_ids = torch.cat([position_ids, teacher_response_position_ids], dim=-1)
```

**返回的batch包含**：
- `input_ids`, `attention_mask`, `position_ids`：学生的prompt+response
- `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`：prompt+教师response
- `responses`：学生生成的响应
- `teacher_response`：教师的响应

---

## 七、训练方法的理论基础

### 7.1 对抗训练框架

GAD采用了类似GAN的对抗训练框架：

**生成器（Generator/Actor）**：
- 目标：生成能够欺骗判别器的高质量响应
- 损失：最大化判别器给学生响应的分数
- 通过PPO算法优化

**判别器（Discriminator/Critic）**：
- 目标：区分教师响应和学生响应
- 损失：`-log(σ(r_teacher - r_student))`
- 期望教师分数高于学生分数

### 7.2 与传统GAN的区别

1. **离散序列生成**：使用PPO而非直接梯度
2. **有监督信号**：有教师响应作为正样本
3. **稳定性改进**：
   - GRPO的组内归一化
   - KL散度约束
   - 梯度裁剪

### 7.3 与SeqKD的对比

**SeqKD（Sequence-level Knowledge Distillation）**：
```
Loss = -log P_student(y_teacher | x)
```
直接最大化学生模型生成教师响应的概率。

**GAD**：
```
Discriminator: max E[log σ(D(x, y_teacher) - D(x, y_student))]
Generator: max E[D(x, y_student)]
```
通过对抗训练让学生学习教师的响应分布。

**优势**：
- 不需要精确匹配教师的token序列
- 允许学生探索不同的表达方式
- 更适合黑盒教师模型

---

## 八、关键超参数分析

### 8.1 采样数量 n=8

```python
actor_rollout_ref.rollout.n=8
```

**作用**：
- GRPO需要多个样本进行组内比较
- n越大，优势估计越准确，但计算成本越高
- 8是经验值，平衡了效果和效率

### 8.2 温度 temperature=0.8

```python
actor_rollout_ref.rollout.temperature=0.8
```

**作用**：
- 控制生成的随机性
- 0.8保证了足够的探索空间
- 太低会导致模式崩溃，太高会导致质量下降

### 8.3 KL系数 kl_coef=0.001

```python
algorithm.kl_ctrl.kl_coef=0.001
actor_rollout_ref.actor.kl_loss_coef=0.001
```

**作用**：
- 防止策略偏离参考策略太远
- 保持生成的稳定性
- 0.001是较小的值，允许较大的策略更新

### 8.4 裁剪范围 clip_ratio=0.2

```python
actor_rollout_ref.actor.clip_ratio=0.2
```

**作用**：
- PPO的核心超参数
- 限制策略更新的幅度
- 0.2是标准值

### 8.5 学习率 lr=1e-6

```python
actor_rollout_ref.actor.optim.lr=1e-6
critic.optim.lr=1e-6
```

**作用**：
- 较小的学习率保证训练稳定
- 因为是从warmup继续训练，模型已经接近收敛
- 对抗训练需要小心调整学习率

## 九、训练监控指标

### 9.1 判别器指标

```python
metrics = {
    "critic/d_loss": d_loss,  # 判别器损失
    "critic/d_acc": d_acc,    # 判别准确率
    "critic/student_value_mean": student_value_mean,  # 学生平均分
    "critic/teacher_value_mean": teacher_value_mean,  # 教师平均分
    "critic/grad_norm": grad_norm,  # 梯度范数
}
```

**理想状态**：
- `d_acc` 应该在0.5-0.8之间（太高说明生成器太弱，太低说明判别器太弱）
- `teacher_value_mean > student_value_mean`（但差距应该逐渐缩小）

### 9.2 生成器指标

```python
metrics = {
    "actor/pg_loss": pg_loss,  # 策略梯度损失
    "actor/pg_clipfrac": pg_clipfrac,  # 裁剪比例
    "actor/ppo_kl": ppo_kl,  # PPO KL散度
    "actor/entropy": entropy,  # 策略熵
    "actor/kl_loss": kl_loss,  # KL损失
    "actor/grad_norm": grad_norm,  # 梯度范数
}
```

**理想状态**：
- `pg_clipfrac` 在0.1-0.3之间（说明策略更新适中）
- `entropy` 保持一定水平（避免模式崩溃）
- `ppo_kl` 较小（策略更新稳定）

---

## 十、总结

### 核心创新点

1. **Critic作为Discriminator**：巧妙地将VeRL的Critic模块改造为判别器
2. **GRPO优势估计**：组内相对比较，不依赖绝对奖励值
3. **黑盒蒸馏**：只需要教师的响应文本，不需要logits
4. **稳定的对抗训练**：通过PPO、KL约束、梯度裁剪保证稳定性

### 训练流程总结

```
For each batch:
  1. 生成：Actor生成n=8个响应
  2. 打分：Discriminator对学生响应打分
  3. 优势：GRPO计算组内相对优势
  4. 更新判别器：区分教师和学生响应
  5. 更新生成器：最大化判别器分数（通过PPO）
```

### 与Warmup阶段的区别

| 维度 | Warmup阶段 | GAD阶段 |
|------|-----------|---------|
| Critic作用 | 预测value | 作为判别器打分 |
| 训练目标 | 模仿教师响应 | 对抗训练 |
| 损失函数 | 交叉熵 | 判别器损失+PPO损失 |
| 优势估计 | GAE | GRPO |
| 探索性 | 较低 | 较高（temperature=0.8） |

这个设计使得GAD能够在黑盒场景下有效地进行知识蒸馏，同时保持训练的稳定性和生成质量。

### 关键技术要点总结

1. **对抗训练架构**：
   - Critic被hack成判别器，需要同时计算teacher和student分数
   - 使用对比损失：`-log(σ(r_teacher - r_student))`
   - 交替更新：先更新判别器，再更新生成器

2. **分数计算机制**：
   - 使用Value Head或logits的某个维度作为分数
   - 只在最后一个有效token输出分数（outcome-based）
   - 通过mask机制确保只有有效token参与计算

3. **GRPO优势估计**：
   - 组内相对比较，每个prompt生成n=8个响应
   - 归一化：`(score - mean) / std`
   - 不依赖绝对奖励值，只关心相对好坏

4. **训练稳定性保证**：
   - KL散度约束防止策略偏离太远
   - 梯度裁剪防止梯度爆炸
   - PPO裁剪机制限制策略更新幅度
   - 熵正则化保持生成多样性

5. **数据流设计**：
   - 数据包含prompt、student_response、teacher_response
   - Rollout阶段扩充数据，包含teacher和student的完整序列
   - 同一batch用于更新Critic和Actor

这种设计巧妙地将GAN的对抗训练思想应用到了序列生成的知识蒸馏任务中，实现了在黑盒场景下的有效知识传递。

---

## 十一、Critic打分与GRPO协同工作机制详解

### 11.1 数据流转换的完整过程

在GAD训练中，Critic的打分和GRPO的优势计算是两个紧密协作的步骤。理解它们如何协同工作是掌握GAD核心机制的关键。

#### 11.1.1 从Critic分数到GRPO优势的数据流

```python
# 步骤1: Critic计算原始分数
values = self.critic_wg.compute_values(batch)  # Critic对student响应打分
reward_tensor = batch.batch["values"]  # (bsz, response_length)

# 步骤2: 分数赋值
batch.batch["token_level_scores"] = reward_tensor  # 原始Critic分数

# 步骤3: 可选的KL惩罚调整
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]  # 直接使用Critic分数

# 步骤4: GRPO优势计算
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,  # 使用GRPO
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
)
```

### 11.2 两种"分数"的本质区别

#### 11.2.1 Critic的分数 (`token_level_scores/token_level_rewards`)

**性质**：
- **作用**：提供每个响应的质量评估
- **格式**：`(batch_size, response_length)` 张量
- **来源**：Critic判别器对student响应的打分
- **特点**：只有最后一个有效token有非零值
- **含义**：绝对质量分数，反映响应的整体质量

**示例**：
```python
# 假设batch_size=8, response_length=10
token_level_scores = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.2],  # 响应1: 分数3.2
    [0, 0, 0, 0, 0, 0, 0, 0, 2.8, 0],  # 响应2: 分数2.8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.8],  # 响应3: 分数3.8
    [0, 0, 0, 0, 0, 0, 0, 2.1, 0, 0],  # 响应4: 分数2.1
    # ... 更多响应
]
```

#### 11.2.2 GRPO的优势 (`advantages`)

**性质**：
- **作用**：指导Actor梯度更新的方向和强度
- **格式**：`(batch_size, response_length)` 张量  
- **来源**：基于Critic分数进行组内相对比较计算得出
- **特点**：归一化后的相对优势值，均值为0
- **含义**：相对优势，表示该响应在同组中的相对好坏

### 11.3 GRPO如何使用Critic分数：详细算法解析

#### 11.3.1 GRPO核心算法步骤

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # 来自Critic的分数
    response_mask: torch.Tensor,
    index: np.ndarray,  # prompt的uid，用于分组
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    # Step 1: 聚合分数 - 将token级分数转为响应级分数
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    
    # Step 2: 按prompt分组
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        # 将响应按prompt分组
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # Step 3: 计算组内统计量
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 只有一个响应的情况（不应该发生）
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # 正常情况：计算组内均值和标准差
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
        
        # Step 4: 归一化优势计算
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # 标准GRPO：z-score归一化
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO：只减均值，不除标准差
                scores[i] = scores[i] - id2mean[index[i]]
        
        # Step 5: 扩展到token维度
        scores = scores.unsqueeze(-1) * response_mask  # (batch_size, response_length)
    
    return scores, scores  # advantages, returns
```

#### 11.3.2 详细的数据流示例

假设我们有2个prompt，每个生成4个响应：

```python
# 原始数据
prompts = ["What is AI?", "Explain ML"]
responses = [
    # Prompt 0的4个响应
    ["AI is artificial intelligence", "AI means smart computers", 
     "AI is machine learning", "AI helps humans"],
    # Prompt 1的4个响应  
    ["ML is machine learning", "ML uses algorithms", 
     "ML learns from data", "ML predicts patterns"]
]

# Step 1: Critic打分 (只有最后token有值)
critic_scores = torch.tensor([
    [0, 0, 0, 3.2],  # 响应1的分数：3.2
    [0, 0, 0, 2.8],  # 响应2的分数：2.8
    [0, 0, 0, 3.8],  # 响应3的分数：3.8
    [0, 0, 0, 2.1],  # 响应4的分数：2.1
    [0, 0, 0, 4.1],  # 响应5的分数：4.1
    [0, 0, 0, 3.9],  # 响应6的分数：3.9
    [0, 0, 0, 3.5],  # 响应7的分数：3.5
    [0, 0, 0, 4.3],  # 响应8的分数：4.3
])

# Step 2: 聚合为响应级分数
response_scores = critic_scores.sum(dim=-1)  # [3.2, 2.8, 3.8, 2.1, 4.1, 3.9, 3.5, 4.3]

# Step 3: 按prompt分组
index = [0, 0, 0, 0, 1, 1, 1, 1]  # prompt的uid
group_0_scores = [3.2, 2.8, 3.8, 2.1]  # Prompt 0的4个响应
group_1_scores = [4.1, 3.9, 3.5, 4.3]  # Prompt 1的4个响应

# Step 4: 计算组内统计量
group_0_mean = (3.2 + 2.8 + 3.8 + 2.1) / 4 = 2.975
group_0_std = std([3.2, 2.8, 3.8, 2.1]) = 0.697

group_1_mean = (4.1 + 3.9 + 3.5 + 4.3) / 4 = 3.95
group_1_std = std([4.1, 3.9, 3.5, 4.3]) = 0.327

# Step 5: 计算GRPO优势
advantages = [
    # Group 0 (相对于组内均值2.975)
    (3.2 - 2.975) / 0.697 = 0.323,   # 响应1优势：正值，好于平均
    (2.8 - 2.975) / 0.697 = -0.251,  # 响应2优势：负值，差于平均
    (3.8 - 2.975) / 0.697 = 1.184,   # 响应3优势：正值，最好
    (2.1 - 2.975) / 0.697 = -1.255,  # 响应4优势：负值，最差
    # Group 1 (相对于组内均值3.95)
    (4.1 - 3.95) / 0.327 = 0.459,    # 响应5优势：正值
    (3.9 - 3.95) / 0.327 = -0.153,   # 响应6优势：负值
    (3.5 - 3.95) / 0.327 = -1.376,   # 响应7优势：负值，最差
    (4.3 - 3.95) / 0.327 = 1.070,    # 响应8优势：正值，最好
]

# Step 6: 扩展到token维度
final_advantages = [
    [0, 0, 0, 0.323],   # 只有最后一个token有优势值
    [0, 0, 0, -0.251],
    [0, 0, 0, 1.184],
    [0, 0, 0, -1.255],
    [0, 0, 0, 0.459],
    [0, 0, 0, -0.153],
    [0, 0, 0, -1.376],
    [0, 0, 0, 1.070],
]
```

### 11.4 Actor梯度更新中的使用

#### 11.4.1 PPO策略梯度计算

最终，Actor使用这些GRPO优势进行PPO更新：

```python
# 在Actor的update_policy方法中
pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
    old_log_prob=old_log_prob,      # 旧策略的log概率
    log_prob=current_log_prob,      # 当前策略的log概率
    advantages=advantages,          # 来自GRPO的优势 ← 关键输入
    response_mask=response_mask,
    cliprange=0.2
)

# PPO损失的核心公式
ratio = torch.exp(log_prob - old_log_prob)  # 重要性采样比率
pg_losses1 = -advantages * ratio            # 策略梯度损失1
pg_losses2 = -advantages * torch.clamp(ratio, 1-0.2, 1+0.2)  # 裁剪版本
pg_loss = torch.max(pg_losses1, pg_losses2).mean()  # 取较大值（更保守）
```

#### 11.4.2 梯度更新的直观理解

```python
# 对于优势为正的响应（好响应）
if advantage > 0:
    # 增加生成该响应的概率
    gradient_direction = "increase_probability"
    
# 对于优势为负的响应（差响应）  
if advantage < 0:
    # 减少生成该响应的概率
    gradient_direction = "decrease_probability"
```

### 11.5 协同工作的关键优势

#### 11.5.1 稳定性保证

1. **组内归一化**：
   - GRPO的组内比较消除了不同batch间的分数差异
   - 减少了训练方差，提高稳定性
   - 每组的优势均值始终为0

2. **相对性原则**：
   - 不依赖Critic分数的绝对值
   - 只关心同组内的相对好坏
   - 避免了奖励尺度问题

#### 11.5.2 效率优势

1. **共享计算**：
   - 同一个batch用于Critic打分和Actor更新
   - 避免重复的前向传播

2. **简化设计**：
   - 不需要复杂的value function训练
   - 避免了传统PPO中的value loss

#### 11.5.3 对抗训练适配

1. **动态适应**：
   - 随着Actor生成质量提升，Critic分数分布会变化
   - GRPO自动适应新的分数分布
   - 保持训练的持续性

2. **探索与利用平衡**：
   - 温度采样保证探索
   - GRPO优势引导利用
   - 两者结合实现最优策略搜索

### 11.6 与传统方法的对比

#### 11.6.1 与标准PPO的区别

**标准PPO**：
```python
# 需要value function预测未来奖励
values = critic(states)
returns = compute_returns(rewards, values, gamma, lam)
advantages = returns - values
```

**GAD + GRPO**：
```python
# 直接使用outcome reward，通过组内比较计算优势
scores = discriminator(responses)
advantages = grpo_normalize(scores, group_indices)
```

#### 11.6.2 优势对比

| 维度 | 标准PPO | GAD + GRPO |
|------|---------|------------|
| Value Function | 需要训练 | 不需要 |
| 奖励类型 | 需要密集奖励 | 只需outcome奖励 |
| 方差 | 较高 | 较低（组内归一化） |
| 计算复杂度 | 高 | 低 |
| 适用场景 | 通用RL | 文本生成蒸馏 |

### 11.7 实际训练中的监控指标

#### 11.7.1 Critic分数监控

```python
# 监控Critic分数的分布
metrics = {
    "critic/score/mean": torch.mean(sequence_score).item(),
    "critic/score/max": torch.max(sequence_score).item(), 
    "critic/score/min": torch.min(sequence_score).item(),
    "critic/score/std": torch.std(sequence_score).item(),
}
```

#### 11.7.2 GRPO优势监控

```python
# 监控GRPO优势的分布
metrics = {
    "critic/advantages/mean": torch.mean(valid_advantages).item(),  # 应该接近0
    "critic/advantages/max": torch.max(valid_advantages).item(),
    "critic/advantages/min": torch.min(valid_advantages).item(),
    "critic/advantages/std": torch.std(valid_advantages).item(),   # 应该接近1
}
```

#### 11.7.3 理想的监控状态

- **Critic分数**：随训练逐渐提升，但组内方差保持稳定
- **GRPO优势**：均值始终接近0，标准差接近1
- **分数差异**：同组内应有明显差异，不同组间可以有差异

这种Critic打分与GRPO协同的设计，巧妙地将判别器的绝对质量评估转换为相对优势信号，为Actor提供了稳定而有效的梯度更新方向，是GAD训练成功的关键机制。

1. **交替更新而非同时更新**:
   - 避免梯度冲突
   - 每次更新都基于固定的对手策略

2. **使用detached的优势函数**:
   - 防止梯度从Actor传回Critic
   - 保持两个网络的独立性

3. **Critic先更新**:
   - 确保Actor使用的奖励信号是最新的
   - Critic作为"裁判"需要先适应当前的生成质量

**效率考虑**:
1. **共享同一个batch**:
   - 减少数据加载开销
   - 充分利用生成的样本

2. **Critic只在最后一个token输出分数**:
   - 减少计算量
   - 符合outcome-based奖励的设定

---

## 五、Critic分数计算原理

### 5.1 Critic模型架构

Critic实际上是一个**完整的语言模型**（如Qwen2.5-7B-Instruct），与Actor使用相同的基础架构。关键区别在于：

- **Actor**：使用语言模型头（lm_head）输出词表概率
- **Critic**：使用语言模型的logits作为分数输出，或使用Value Head

### 5.2 分数计算的完整流程

#### 5.2.1 模型前向传播

```python
def _forward_micro_batch(self, micro_batch, compute_teacher):
    # Step 1: 确定处理的是teacher还是student响应
    if compute_teacher:
        response_length = micro_batch["teacher_response"].size(-1)
        input_ids = micro_batch["teacher_input_ids"]  # prompt + teacher_response
        attention_mask = micro_batch["teacher_attention_mask"]
        position_ids = micro_batch["teacher_position_ids"]
    else:
        response_length = micro_batch["responses"].size(-1)
        input_ids = micro_batch["input_ids"]  # prompt + student_response
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
    
    batch, seqlen = input_ids.shape  # 例如：(4, 512)
    
    # Step 2: 模型前向传播
    with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
        output = self.critic_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        
        # 获取values（可能是logits或value head的输出）
        if hasattr(self.critic_module, "v_head"):
            # 使用trl的AutoModelForCausalLMWithValueHead
            values = output[2]  # (batch_size, seq_len, 1)
        else:
            # 标准语言模型，使用logits的某个维度
            values = output.logits  # (batch_size, seq_len, vocab_size)
            values = values.squeeze(-1)  # 可能需要降维处理
```

#### 5.2.2 提取响应部分的分数

```python
# 只保留响应部分的logits/values
values = values[:, -response_length:]  # (batch_size, response_length)
```

#### 5.2.3 关键Hack：只保留最后一个有效token的分数

```python
# 计算响应的mask
response_mask = attention_mask[:, -response_length:]  # (batch_size, response_length)

# 找到每个序列的最后一个有效token位置
response_lengths = response_mask.sum(dim=1).long()  # (batch_size,)
last_token_indices = response_lengths - 1  # (batch_size,)

# 创建一个mask，只在最后一个有效token位置为True
last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
last_token_mask[batch_indices, last_token_indices] = True
# last_token_mask shape: (batch_size, response_length)
# 例如：[[False, False, False, True],   # 第4个token是最后一个有效token
#        [False, False, True, False]]   # 第3个token是最后一个有效token

# 只保留最后一个token的分数，其他位置置零
values = values * last_token_mask.type_as(values)
# values shape: (batch_size, response_length)
# 但只有last_token位置有非零值
```

### 5.3 分数的实际含义与聚合

#### 5.3.1 Value Head架构（推荐）

```python
from trl import AutoModelForCausalLMWithValueHead

# 在模型初始化时
critic_module = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
```

**Value Head结构**:
```
Language Model (Qwen2.5-7B)
    ↓
hidden_states (batch, seq_len, hidden_dim)
    ↓
Value Head: Linear(hidden_dim, 1)
    ↓
values (batch, seq_len, 1)
```

这样每个token位置输出一个标量分数。

#### 5.3.2 完整的分数计算示例

```python
# 输入数据
prompt = "What is the capital of France?"
student_response = "Paris is capital"
teacher_response = "The capital of France is Paris"

# Tokenize
student_input_ids = tokenize(prompt + student_response)  # [101, 234, ..., 567]
teacher_input_ids = tokenize(prompt + teacher_response)  # [101, 234, ..., 789]

# 计算student分数
student_output = critic_module(student_input_ids)
student_values = student_output[2]  # (1, seq_len, 1) 如果有value head
# 例如：[[0.1, 0.2, 0.3, 0.5, 1.2, 2.3, 0, 0]]
#                              ↑ 最后一个有效token

# 只保留最后一个有效token
response_length = len(tokenize(student_response))
student_values = student_values[:, -response_length:]  # (1, 4, 1)
# [[1.2, 2.3, 0, 0]]

# 应用last_token_mask
last_token_mask = [[0, 1, 0, 0]]  # 假设第2个是最后有效token
student_values = student_values * last_token_mask  # [[0, 2.3, 0, 0]]

# 计算teacher分数（同样的流程）
teacher_output = critic_module(teacher_input_ids)
teacher_values = teacher_output[2][:, -response_length:]
# 假设：[[0, 0, 0, 4.8]]  # 最后一个token的分数是4.8

# 在损失函数中求和
student_reward = student_values.sum(dim=1)  # [2.3]
teacher_reward = teacher_values.sum(dim=1)  # [4.8]

# 计算损失
d_loss = -log(sigmoid(4.8 - 2.3)) = -log(sigmoid(2.5)) ≈ 0.08
```

### 5.4 为什么只用最后一个token的分数？

#### 5.4.1 设计原因

1. **Outcome-based奖励**：
   - GAD使用的是结果级别的奖励，不是过程级别
   - 整个响应的质量用一个标量表示
   - 类似于人类评判：看完整个回答后给出总体评分

2. **计算效率**：
   - 只需要计算最后一个token的value
   - 可以使用因果注意力机制
   - 最后一个token包含了整个序列的信息

3. **与GRPO配合**：
   - GRPO需要每个响应的总分数（标量）
   - 不需要token级别的细粒度分数

#### 5.4.2 与传统Critic的对比

**传统PPO Critic**：
```python
# 每个token都有value，预测未来累积奖励
values = [v1, v2, v3, v4, v5]  # 每个token的value
returns = [r1, r2, r3, r4, r5]  # 每个token的实际回报
loss = MSE(values, returns)
```

**GAD Discriminator**：
```python
# 只有最后一个token有分数，表示整体质量
student_score = [0, 0, 0, 0, 2.3]  # 只有最后一个有值
teacher_score = [0, 0, 0, 0, 4.8]
loss = -log(sigmoid(4.8 - 2.3))
```

## 十二、Actor模型GRPO参数更新详解

### 12.1 GRPO组内优势计算的数学原理

#### 12.1.1 核心思想

GRPO (Group Relative Policy Optimization) 的核心是**组内相对比较**，而不是绝对评分：

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6):
    """
    Args:
        token_level_rewards: (bs, response_length) - Critic给出的token级分数
        response_mask: (bs, response_length) - 有效token的mask
        index: (bs,) - 每个样本所属的prompt组ID
    """
    # Step 1: 计算每个响应的总分数
    scores = token_level_rewards.sum(dim=-1)  # (bs,) 每个响应的总分
    
    # Step 2: 按prompt分组
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])  # 同一prompt的所有响应分数
    
    # Step 3: 计算组内统计量
    for idx in id2score:
        if len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))  # 组内均值
            id2std[idx] = torch.std(torch.tensor(id2score[idx]))    # 组内标准差
        else:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
    
    # Step 4: 计算相对优势
    for i in range(bsz):
        # 标准化：(分数 - 组内均值) / 组内标准差
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    
    # Step 5: 扩展到token级别
    advantages = scores.unsqueeze(-1) * response_mask  # (bs, response_length)
    
    return advantages, advantages  # advantages和returns相同
```

#### 12.1.2 数学公式

对于同一个prompt的n个响应 $\{r_1, r_2, ..., r_n\}$：

$$\text{Advantage}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

其中：
- $\bar{r} = \frac{1}{n}\sum_{j=1}^n r_j$ （组内均值）
- $\sigma_r = \sqrt{\frac{1}{n}\sum_{j=1}^n (r_j - \bar{r})^2}$ （组内标准差）
- $\epsilon = 1e-6$ （数值稳定性）

#### 12.1.3 具体示例

```python
# 假设有2个prompt，每个prompt生成4个响应
prompt_1_scores = [3.2, 4.1, 2.8, 3.9]  # prompt 1的4个响应分数
prompt_2_scores = [5.1, 4.8, 5.3, 4.9]  # prompt 2的4个响应分数

# 组内归一化
prompt_1_mean = 3.5, prompt_1_std = 0.54
prompt_1_advantages = [(3.2-3.5)/0.54, (4.1-3.5)/0.54, (2.8-3.5)/0.54, (3.9-3.5)/0.54]
                    = [-0.56, 1.11, -1.30, 0.74]

prompt_2_mean = 5.0, prompt_2_std = 0.21  
prompt_2_advantages = [(5.1-5.0)/0.21, (4.8-5.0)/0.21, (5.3-5.0)/0.21, (4.9-5.0)/0.21]
                    = [0.48, -0.95, 1.43, -0.48]
```

### 12.2 KL散度对齐机制

#### 12.2.1 KL惩罚的作用

KL散度对齐确保Actor不会偏离参考策略太远，维持训练稳定性：

```python
def apply_kl_penalty(token_level_scores, log_probs, ref_log_probs, response_mask, kl_coef):
    """
    Args:
        token_level_scores: Critic给出的原始分数
        log_probs: 当前策略的对数概率
        ref_log_probs: 参考策略的对数概率
        kl_coef: KL惩罚系数
    """
    # 计算KL散度
    kl_divergence = log_probs - ref_log_probs  # (bs, response_length)
    
    # 应用KL惩罚
    token_level_rewards = token_level_scores - kl_coef * kl_divergence
    
    return token_level_rewards
```

#### 12.2.2 两种KL对齐方式

**方式1：奖励中的KL惩罚**
```python
# 在ray_trainer.py中
if self.config.kl_penalty_coef > 0:
    token_level_rewards = self.apply_kl_penalty(
        token_level_scores=token_level_scores,
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        response_mask=response_mask,
        kl_coef=self.config.kl_penalty_coef
    )
```

**方式2：损失函数中的KL损失**
```python
# 在dp_actor.py中
if self.config.use_kl_loss:
    ref_log_prob = data["ref_log_prob"]
    # 计算KL损失
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # 添加到策略损失中
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

#### 12.2.3 KL散度的数学意义

$$D_{KL}(P_{current} || P_{ref}) = \sum_t \log P_{current}(a_t|s_t) - \log P_{ref}(a_t|s_t)$$

- **正值**：当前策略比参考策略更倾向于选择该action
- **负值**：当前策略比参考策略更不倾向于选择该action
- **KL惩罚**：防止策略变化过大，保持训练稳定

### 12.3 PPO策略梯度更新的具体实现

#### 12.3.1 PPO损失函数

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, cliprange=0.2):
    """
    PPO的核心：裁剪策略梯度
    """
    # Step 1: 计算重要性采样比率
    ratio = torch.exp(log_prob - old_log_prob)  # π_θ(a|s) / π_θ_old(a|s)
    
    # Step 2: 计算未裁剪的策略损失
    pg_losses1 = -advantages * ratio  # -A * π_θ/π_θ_old
    
    # Step 3: 计算裁剪的策略损失
    pg_losses2 = -advantages * torch.clamp(ratio, 1-cliprange, 1+cliprange)
    
    # Step 4: 取最大值（最保守的更新）
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    # Step 5: 聚合损失
    pg_loss = masked_mean(pg_losses, response_mask)
    
    return pg_loss
```

#### 12.3.2 双重裁剪机制

GAD实现了更复杂的双重裁剪：

```python
# 标准PPO裁剪
pg_losses2 = -advantages * torch.clamp(ratio, 1-cliprange, 1+cliprange)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

# 额外的下界裁剪（对负优势）
pg_losses3 = -advantages * clip_ratio_c  # clip_ratio_c = 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

# 最终损失
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

#### 12.3.3 策略梯度的直观理解

```python
# 策略梯度的本质
∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]

# PPO的改进：限制策略变化幅度
∇_θ J_PPO(θ) = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

# 其中：
# - ratio = π_θ(a|s) / π_θ_old(a|s)：新旧策略比率
# - A(s,a)：优势函数（来自GRPO）
# - clip(ratio, 1-ε, 1+ε)：裁剪比率，防止更新过大
```

### 12.4 Actor参数更新的完整数据流

#### 12.4.1 数据流概览

```python
def actor_update_pipeline():
    # 1. 数据准备
    prompts = load_prompts()
    
    # 2. 生成响应
    responses = actor.generate(prompts, n=8)  # 每个prompt生成8个响应
    
    # 3. 计算旧策略概率
    old_log_probs = actor.compute_log_probs(prompts, responses)
    
    # 4. Critic评分
    token_level_scores = critic.compute_values(prompts, responses)
    
    # 5. KL惩罚（可选）
    if use_kl_penalty:
        ref_log_probs = ref_model.compute_log_probs(prompts, responses)
        token_level_rewards = apply_kl_penalty(token_level_scores, old_log_probs, ref_log_probs)
    else:
        token_level_rewards = token_level_scores
    
    # 6. GRPO优势计算
    advantages = compute_grpo_advantage(token_level_rewards, response_mask, prompt_indices)
    
    # 7. Actor更新
    for epoch in range(ppo_epochs):
        for mini_batch in data_loader:
            # 7.1 前向传播
            new_log_probs = actor.forward(mini_batch)
            
            # 7.2 计算PPO损失
            pg_loss = compute_policy_loss(old_log_probs, new_log_probs, advantages)
            
            # 7.3 添加熵正则化
            if entropy_coeff > 0:
                entropy_loss = compute_entropy_loss(new_log_probs)
                total_loss = pg_loss - entropy_coeff * entropy_loss
            
            # 7.4 添加KL损失（可选）
            if use_kl_loss:
                kl_loss = compute_kl_loss(new_log_probs, ref_log_probs)
                total_loss += kl_loss_coeff * kl_loss
            
            # 7.5 反向传播
            total_loss.backward()
            
            # 7.6 梯度裁剪
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            
            # 7.7 参数更新
            optimizer.step()
            optimizer.zero_grad()
```

### 12.5 GRPO的核心优势和原理

#### 12.5.1 相对性原则

```python
# 传统PPO：绝对优势
traditional_advantage = reward - baseline  # 可能受batch质量影响

# GRPO：相对优势
grpo_advantage = (reward - group_mean) / group_std  # 组内相对比较
```

**优势**：
- **消除batch偏差**：不同batch的绝对分数可能不同，但相对排序稳定
- **自适应归一化**：自动适应不同难度的prompt
- **稳定训练**：减少训练方差，提高收敛稳定性

#### 12.5.2 与传统PPO的对比

| 维度 | 传统PPO | GRPO |
|------|---------|------|
| **优势计算** | reward - baseline | (reward - group_mean) / group_std |
| **参考点** | 全局baseline | 组内均值 |
| **归一化** | 无或全局归一化 | 组内标准化 |
| **稳定性** | 受baseline质量影响 | 相对稳定 |
| **适应性** | 固定baseline | 自适应组内比较 |
| **方差** | 较高 | 较低 |

这样，Actor模型通过GRPO的组内优势计算、KL散度对齐和PPO策略梯度更新，实现了稳定有效的参数更新，不断提升生成质量以"欺骗"Critic判别器，从而在对抗训练中获得更好的性能