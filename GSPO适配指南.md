# GSPO 算法适配指南

## 一、GSPO 概述

### 1.1 什么是 GSPO？

**GSPO (Group-wise Sequence-level Policy Optimization)** 是一种改进的策略优化算法，论文见：https://arxiv.org/pdf/2507.18071

### 1.2 GSPO vs 标准 PPO

| 维度 | 标准 PPO | GSPO |
|------|---------|------|
| **重要性采样比率** | Token 级：`π_θ(y_t) / π_old(y_t)` | 序列级：`[π_θ(y) / π_old(y)]^(1/\|y\|)` |
| **优势估计** | GAE（Token 级） | GRPO（序列级） |
| **损失聚合** | `token-mean` | `seq-mean-token-mean` |
| **适用场景** | 通用 RL | 序列级奖励（如代码生成、数学推理） |

### 1.3 GSPO 的核心创新

**序列级重要性采样比率**：
```
s_i(θ) = [π_θ(y_i|x) / π_old(y_i|x)]^(1/|y_i|)
       = exp[(1/|y_i|) * Σ_t log(π_θ(y_i,t) / π_old(y_i,t))]
```

**组合比率**（用于梯度计算）：
```
s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t) / sg[π_θ(y_i,t)]
```
其中 `sg` 表示 stop gradient。

---

## 二、当前框架的 GSPO 支持情况

### 2.1 已有的 GSPO 实现

好消息：**新的 verl 框架已经内置了 GSPO 支持**！

在 `verl/trainer/ppo/core_algos.py` 中：

```python
@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.
    """
    # 计算序列级重要性采样比率
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    
    # 组合比率（token 级）
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    
    # PPO clipping
    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    # 序列级聚合
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info
    )
    
    return pg_loss, pg_metrics
```

### 2.2 GSPO 的优势估计

GSPO 通常与 **GRPO** 优势估计配合使用：

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GRPO 优势：同一 prompt 的多个回复进行组内标准化
    """
    scores = token_level_rewards.sum(dim=-1)  # 序列级分数
    
    # 按 uid 分组
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    # 计算组内均值和标准差
    id2mean = {idx: torch.mean(torch.tensor(id2score[idx])) for idx in id2score}
    id2std = {idx: torch.std(torch.tensor(id2score[idx])) for idx in id2score}
    
    # 标准化优势
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    
    # 广播到 token 级别
    advantages = scores.unsqueeze(-1) * response_mask
    return advantages, advantages
```

---

## 三、如何启用 GSPO

### 3.1 配置修改

要使用 GSPO，只需修改配置文件中的两个参数：

```yaml
# 1. 设置优势估计器为 GRPO
algorithm:
  adv_estimator: grpo  # 或 grpo_vectorized（更快）
  norm_adv_by_std_in_grpo: true  # 是否标准化（推荐 true）

# 2. 设置策略损失为 GSPO
actor_rollout_ref:
  actor:
    policy_loss: gspo  # 从 "ppo" 改为 "gspo"
    loss_agg_mode: seq-mean-token-mean  # GSPO 推荐的聚合模式
    clip_ratio: 0.2  # 裁剪参数
    clip_ratio_low: 0.2  # 可选：不对称裁剪
    clip_ratio_high: 0.2
```

### 3.2 完整配置示例

```yaml
# 算法配置
algorithm:
  adv_estimator: grpo  # 使用 GRPO 优势估计
  norm_adv_by_std_in_grpo: true
  gamma: 1.0
  lam: 0.95
  use_kl_in_reward: false

# Actor 配置
actor_rollout_ref:
  rollout:
    n: 8  # 每个 prompt 生成 8 个回复（GRPO 需要多个样本）
    temperature: 0.8
    top_p: 0.9
    
  actor:
    policy_loss: gspo  # 🔥 使用 GSPO 策略损失
    loss_agg_mode: seq-mean-token-mean  # 🔥 序列级聚合
    clip_ratio: 0.2
    optim:
      lr: 1e-6
    
  ppo_mini_batch_size: 256
  ppo_micro_batch_size_per_gpu: 8
  ppo_epochs: 1

# Critic 配置（如果使用）
critic:
  optim:
    lr: 1e-6
  ppo_epochs: 1
```

---

## 四、GSPO + GAD 的组合

如果你想将 **GSPO** 与 **GAD（判别器训练）** 结合，需要以下修改：

### 4.1 整体架构

```
数据加载 (包含 teacher_response)
  ↓
Actor 生成学生回复 (n=8)
  ↓
判别器评分（学生 vs 教师）
  ↓
GRPO 计算优势（组内标准化）
  ↓
更新判别器（判别器损失）
  ↓
更新 Actor（GSPO 损失）
```

### 4.2 关键修改点

#### 1. Critic 使用判别器损失（已在前面的分析文档中详述）

```python
# verl/workers/critic/dp_critic.py
def update_critic(self, data: DataProto):
    # 双路前向推理
    student_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=False)
    teacher_vpreds = self._forward_micro_batch(micro_batch, compute_teacher=True)
    
    # 判别器损失
    d_loss = core_algos.compute_discriminator_loss(
        student_vpreds=student_vpreds,
        teacher_vpreds=teacher_vpreds,
        response_mask=response_mask,
        teacher_response_mask=teacher_response_mask,
    )
```

#### 2. Actor 使用 GSPO 损失

```python
# 配置文件
actor_rollout_ref:
  actor:
    policy_loss: gspo  # 使用 GSPO
    loss_agg_mode: seq-mean-token-mean
```

#### 3. 优势估计使用 GRPO

```python
# 配置文件
algorithm:
  adv_estimator: grpo  # 使用 GRPO
```

### 4.3 完整的 GSPO + GAD 配置

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
  critic_warmup: 10  # 前 10 步只训练判别器
  total_epochs: 2
  save_freq: 50
```

---

## 五、GSPO vs PPO 的性能对比

### 5.1 适用场景

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| **序列级奖励** | GSPO | 更好地处理稀疏奖励 |
| **代码生成** | GSPO | Pass@k 等序列级指标 |
| **数学推理** | GSPO | 最终答案正确性 |
| **对话生成** | PPO 或 GSPO | 都可以 |
| **Token 级奖励** | PPO | 更自然 |

### 5.2 GSPO 的优势

1. **更稳定的训练**：序列级重要性采样减少方差
2. **更好的序列级优化**：直接优化序列质量
3. **与 GRPO 天然配合**：都是序列级方法

### 5.3 GSPO 的劣势

1. **需要多个样本**：`rollout.n` 通常需要 ≥ 4
2. **计算开销略大**：需要计算序列级比率
3. **不适合 token 级奖励**：如果奖励是 token 级的，PPO 更合适

---

## 六、实现检查清单

### 6.1 启用 GSPO（不含 GAD）

- [ ] 修改配置：`algorithm.adv_estimator = grpo`
- [ ] 修改配置：`actor_rollout_ref.actor.policy_loss = gspo`
- [ ] 修改配置：`actor_rollout_ref.actor.loss_agg_mode = seq-mean-token-mean`
- [ ] 确保 `rollout.n ≥ 4`（GRPO 需要多个样本）
- [ ] 验证训练指标：`actor/pg_clipfrac`, `actor/ppo_kl`

### 6.2 启用 GSPO + GAD

**前提**：完成 GAD 代码适配（见 `GAD代码适配分析.md`）

- [ ] 修改 `dp_critic.py`：支持判别器训练
- [ ] 添加 `compute_discriminator_loss` 到 `core_algos.py`
- [ ] 修改 `rl_dataset.py`：加载 `teacher_response`
- [ ] 修改配置：启用 GSPO（见上）
- [ ] 修改配置：`trainer.critic_warmup = 10`
- [ ] 验证判别器指标：`critic/d_loss`, `critic/d_acc`
- [ ] 验证 Actor 指标：`actor/pg_clipfrac`, `actor/ppo_kl`

---

## 七、代码修改示例

### 7.1 不需要修改代码！

如果只是启用 GSPO（不含 GAD），**不需要修改任何代码**，只需修改配置文件：

```bash
# 原始启动命令（使用 PPO）
python -m verl.trainer.main_ppo \
  --config-path configs \
  --config-name ppo_config \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.actor.policy_loss=ppo

# 修改为 GSPO
python -m verl.trainer.main_ppo \
  --config-path configs \
  --config-name ppo_config \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.policy_loss=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
```

### 7.2 如果需要 GSPO + GAD

需要先完成 GAD 适配（见 `GAD代码适配分析.md`），然后：

```bash
python -m verl.trainer.main_ppo \
  --config-path configs \
  --config-name gad_gspo_config \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.policy_loss=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  trainer.critic_warmup=10 \
  data.train_files=/path/to/data_with_teacher_response.parquet
```

---

## 八、调试和验证

### 8.1 关键指标

**GSPO 特有指标**：
- `actor/pg_clipfrac`：裁剪比例（应在 0.1-0.3）
- `actor/ppo_kl`：KL 散度（应保持较小）
- `actor/entropy`：策略熵（不应过快下降）

**GRPO 特有指标**：
- 优势的均值应接近 0（组内标准化）
- 优势的标准差应接近 1（如果 `norm_adv_by_std_in_grpo=true`）

**GAD 特有指标**（如果使用）：
- `critic/d_loss`：判别器损失（应逐渐下降）
- `critic/d_acc`：判别准确率（应从 0.5 上升到 > 0.7）
- `critic/student_value_mean`：学生得分（应逐渐上升）
- `critic/teacher_value_mean`：教师得分（应保持稳定且较高）

### 8.2 常见问题

**问题 1**：`pg_clipfrac` 过高（> 0.5）
- **原因**：学习率过大或 `clip_ratio` 过小
- **解决**：降低学习率或增大 `clip_ratio`

**问题 2**：`d_acc` 不上升（停留在 0.5）
- **原因**：判别器训练失败
- **解决**：检查 `teacher_response` 是否正确加载，增加 `critic_warmup`

**问题 3**：训练不稳定
- **原因**：`rollout.n` 过小
- **解决**：增大 `rollout.n` 到 8 或 16

---

## 九、总结

### 9.1 GSPO 的核心要点

1. **序列级重要性采样**：更适合序列级奖励
2. **与 GRPO 配合**：组内标准化优势
3. **配置简单**：只需修改两个参数
4. **无需代码修改**：框架已内置支持

### 9.2 GSPO + GAD 的组合

- **Critic**：使用判别器损失（对抗训练）
- **Actor**：使用 GSPO 损失（序列级优化）
- **优势估计**：使用 GRPO（组内标准化）
- **适用场景**：需要教师指导的序列级任务

### 9.3 推荐配置

**纯 GSPO（无 GAD）**：
```yaml
algorithm.adv_estimator: grpo
actor_rollout_ref.actor.policy_loss: gspo
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean
actor_rollout_ref.rollout.n: 8
```

**GSPO + GAD**：
```yaml
# 在上述基础上添加：
trainer.critic_warmup: 10
data.train_files: /path/to/data_with_teacher_response.parquet
# 并完成 dp_critic.py 的 GAD 适配
```

---

## 十、参考资料

- **GSPO 论文**：https://arxiv.org/pdf/2507.18071
- **GRPO 论文**：https://arxiv.org/abs/2402.03300
- **verl 框架文档**：https://github.com/volcengine/verl
- **GAD 适配分析**：见 `GAD代码适配分析.md`
