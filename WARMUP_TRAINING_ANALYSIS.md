# Warm-up 阶段训练流程分析

## 概述

本文档分析 `scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh` 启动脚本的实际训练逻辑，验证其是否符合 GAD (Generative Adversarial Distillation) 的 warm-up 阶段伪代码。

## 目标伪代码

```
Warmup Stage:
for each batch (x, yt) ∼ T do
    Update generator G with cross-entropy loss on yt
    Update discriminator D with Bradley-Terry loss
end for
```

## 结论

**✅ 实际实现与伪代码完全匹配**

---

## 详细分析

### 1. 配置参数解析

| 关键参数 | 值 | 作用 |
|---------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 使用 GRPO 算法估计优势函数 |
| `trainer.critic_warmup` | `10` | 前 10 步只更新 Critic，不更新 Actor |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` (默认) | Actor 使用 SFT loss (cross-entropy) |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | 启用 KL 散度约束 |

### 2. Generator (Actor) 更新机制

**Loss 类型**: Cross-Entropy Loss (SFT Loss)

**代码位置**: `verl/verl/workers/actor/dp_actor.py`

```python
if self.config.policy_loss.loss_mode == "vanilla":
    teacher_pg_loss = compute_sft_loss(
        log_prob=teacher_log_prob,
        response_mask=teacher_response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    pg_loss = teacher_pg_loss
```

**Loss 计算**: `verl/verl/trainer/ppo/core_algos.py`

```python
def compute_sft_loss(log_prob, response_mask, loss_agg_mode="token-mean"):
    pg_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss
```

**说明**: `-log_prob` 即为标准的 cross-entropy loss，在 teacher response (yt) 上计算。

### 3. Discriminator (Critic) 更新机制

**Loss 类型**: Bradley-Terry Loss

**代码位置**: `verl/verl/trainer/ppo/core_algos.py`

```python
def compute_discriminator_loss(student_vpreds, teacher_vpreds, response_mask, teacher_response_mask):
    teacher_reward = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
    student_reward = torch.sum(student_vpreds * response_mask, dim=-1)
    d_loss = -nn.functional.logsigmoid(teacher_reward - student_reward).mean()
    return d_loss
```

**说明**: 这是标准的 Bradley-Terry pairwise preference loss：
- `logsigmoid(r_teacher - r_student)` 表示 teacher 优于 student 的概率
- 最小化 `-logsigmoid(...)` 即最大化 teacher 被判定为更优的概率

### 4. Warmup 控制逻辑

**代码位置**: `verl/verl/trainer/ppo/ray_trainer.py`

```python
# 每个 step 都更新 Critic
with marked_timer("update_critic", timing_raw, color="pink"):
    critic_output = self.critic_wg.update_critic(batch)

# 只有当 global_steps > critic_warmup 时才更新 Actor
if self.config.trainer.critic_warmup <= self.global_steps:
    with marked_timer("update_actor", timing_raw, color="red"):
        actor_output = self.actor_rollout_wg.update_actor(batch)
```

**说明**: 
- `critic_warmup=10` 表示前 10 步只更新 Discriminator
- 第 11 步开始同时更新 Generator 和 Discriminator

---

## 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Warm-up Training Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  for each batch (x, yt) in dataset:                         │
│                                                              │
│    ┌──────────────────────────────────────────────────────┐ │
│    │ 1. Generate student response                          │ │
│    │    student_response = G.generate(x)                   │ │
│    └──────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│    ┌──────────────────────────────────────────────────────┐ │
│    │ 2. Compute rewards via Discriminator                  │ │
│    │    r_student = D(x, student_response)                 │ │
│    │    r_teacher = D(x, yt)                               │ │
│    └──────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│    ┌──────────────────────────────────────────────────────┐ │
│    │ 3. Update Discriminator (每步都执行)                   │ │
│    │    d_loss = -logsigmoid(r_teacher - r_student)        │ │
│    │    D.backward(d_loss)                                 │ │
│    └──────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│    ┌──────────────────────────────────────────────────────┐ │
│    │ 4. Update Generator (step > critic_warmup 时执行)     │ │
│    │    g_loss = -log P(yt | x)  # Cross-Entropy           │ │
│    │    G.backward(g_loss)                                 │ │
│    └──────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 对比总结

| 伪代码描述 | 实际实现 | 匹配状态 |
|-----------|----------|---------|
| Generator 用 cross-entropy loss 在 yt 上更新 | `compute_sft_loss(-log_prob)` 在 teacher_response 上计算 | ✅ 完全匹配 |
| Discriminator 用 Bradley-Terry loss 更新 | `compute_discriminator_loss` 使用 `-logsigmoid(r_t - r_s)` | ✅ 完全匹配 |
| Warmup 阶段先只训练 D | `critic_warmup=10` 控制前 10 步只更新 Critic | ✅ 完全匹配 |

---

## 配置默认值来源

`policy_loss.loss_mode` 的默认值定义在 `verl/verl/trainer/config/ppo_trainer.yaml`:

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      # Loss function mode: vanilla / clip-cov / kl-cov
      loss_mode: "vanilla"
```

由于启动脚本未覆盖此参数，因此使用默认值 `"vanilla"`，对应 SFT (cross-entropy) loss。

---

## 关键文件索引

| 文件路径 | 功能 |
|---------|------|
| `scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh` | 启动脚本 |
| `verl/verl/trainer/main_ppo.py` | 训练入口 |
| `verl/verl/trainer/ppo/ray_trainer.py` | 训练循环 & warmup 控制 |
| `verl/verl/trainer/ppo/core_algos.py` | Loss 函数实现 |
| `verl/verl/workers/actor/dp_actor.py` | Actor 更新逻辑 |
| `verl/verl/workers/critic/dp_critic.py` | Critic 更新逻辑 |
| `verl/verl/trainer/config/ppo_trainer.yaml` | 默认配置 |
