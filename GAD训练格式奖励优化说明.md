# GAD 训练格式奖励优化说明

## 修改概述

本次修改在 GAD 训练中加入了格式奖励机制（通用版 v4），核心设计以 ground_truth 为参照，检测格式一致性。

主要惩罚以下问题：
1. **JSON 格式问题**：缺失、不完整、无效、前缀污染、字段缺失
2. **语言污染**：英文思考泄露、中英混杂、JSON 值中的英文句子
3. **内容问题**：连续重复、n-gram 重复、双重输出、时间戳泄露、长度异常

## 核心设计理念

### 以 ground_truth 为参照
- 只有当 ground_truth 是有效 JSON 时，才要求 solution 也是 JSON
- 避免纯文本中恰好包含 `{` 字符导致的误判
- 检查 JSON 字段一致性（solution 是否包含 ground_truth 的所有字段）

### 多问题累加惩罚
- 格式、语言、内容三类问题可以同时存在并累加惩罚
- 增加惩罚力度，防止 reward hacking
- 即使 JSON 正确，也检测重复和语言问题

## 数值设计说明

### Discriminator Value 的典型范围
- Critic 模型输出的 value 典型范围：**[-2, 2]**
- 在 GAD 模式下，只保留最后一个 token 的 value

### 格式奖励的数值范围
- 原始范围：**[-1.5, 0.1]**（主要是惩罚，正向奖励很小）
- 配合 `format_weight=0.3` 后的实际影响：**[-0.45, 0.03]**

### 惩罚项详细说明

#### 1. JSON 格式问题（format）

| 问题类型 | 惩罚值 | 说明 |
|---------|-------|------|
| `json_missing` | -0.5 | 完全没有 JSON 结构 |
| `json_incomplete` | -0.3 | 括号不完整（`{` 多于 `}`） |
| `json_invalid` | -0.25 | JSON 解析失败 |
| `json_prefix` | -0.3 | JSON 前有超过 5 字符的前缀 |
| `json_keys_missing` | -0.2 | 缺少 ground_truth 中的字段 |

#### 2. 语言污染问题（language）

| 问题类型 | 惩罚值 | 说明 |
|---------|-------|------|
| `thinking_leak` | -0.4 | 英文思考泄露（如 "here is", "based on", "let me"） |
| `mixed_language` | -0.4 | 中英文混杂（中文后跟多个英文单词） |
| `json_value_pollution` | -0.35 | JSON 值中包含英文句子 |

#### 3. 内容问题（content）

| 问题类型 | 惩罚值 | 说明 |
|---------|-------|------|
| `repetition_consecutive` | -0.5 | 连续重复（10+ 字符重复 3 次以上） |
| `repetition_ngram` | 动态 | n-gram 重复率 > 35%，惩罚 = (ratio - 0.35) * 0.8，最多 -0.4 |
| `double_output` | -0.35 | JSON 前有超过 50 字符的大段文本 |
| `timestamp_leak` | -0.3 | 时间戳泄露（如 `[2024-01-01 12:00:00]`） |
| `too_long` | 动态 | 输出过长（> 1.5x），惩罚 = (ratio - 1.5) * 0.2，最多 -0.6 |
| `too_short` | -0.3 | 输出过短（< 0.3x） |

#### 4. JSON 内部重复（json_repetition）

| 问题类型 | 惩罚值 | 说明 |
|---------|-------|------|
| `json_repetition` | 动态 | JSON 值的 n-gram 重复率 > 40%，惩罚 = (ratio - 0.4) * 1.0，最多 -0.5 |

#### 5. 奖励项

| 奖励类型 | 奖励值 | 说明 |
|---------|-------|------|
| JSON 正确 | +0.05 | JSON 解析成功且字段完整 |

### 数值影响分析

| 场景 | Discriminator Value | Format Reward | 组合后 (weight=0.3) |
|-----|---------------------|---------------|---------------------|
| 正常输出（JSON 正确） | ~1.0 | +0.05 | 1.015 |
| JSON 缺失 | ~1.0 | -0.5 | 0.85 |
| JSON 不完整 | ~1.0 | -0.3 | 0.91 |
| 英文思考泄露 | ~1.0 | -0.4 | 0.88 |
| 连续重复 | ~1.0 | -0.5 | 0.85 |
| 多问题组合 | ~1.0 | -1.0 ~ -1.5 | 0.55 ~ 0.7 |

**结论**：格式奖励不会抵消 discriminator 的效果，只是提供适度的格式约束。

## 修改的文件

### 1. `verl/verl/utils/reward_score/gad_format_reward.py` (新建)

格式奖励计算函数，核心函数说明：

```python
def compute_format_score(solution_str: str, ground_truth: str = "") -> Dict[str, Any]:
    """
    计算格式奖励分数（通用版 v4）
    
    核心设计：
    - 以 ground_truth 为参照
    - 多个问题可以累加惩罚
    - 增加惩罚力度，防止 reward hacking
    
    Returns:
        {"score": float, "penalties": dict}
        score 范围: [-1.5, 0.1]
    """
```

主要检测函数：
- `check_json_consistency()`: JSON 格式一致性检测
- `check_language_pollution()`: 语言污染检测
- `check_content_issues()`: 内容问题检测
- `compute_ngram_repetition()`: n-gram 重复率计算


### 2. `verl/verl/trainer/ppo/ray_trainer.py` (修改)

在 GAD 训练循环中集成格式奖励：

```python
# 启用格式奖励时，计算格式分数
use_format_reward = self.config.trainer.get("use_format_reward", False)
if use_format_reward and self.use_critic and not self.use_rm:
    # 解码 response 文本
    # 计算格式分数
    # 将格式奖励应用到最后一个 token
    
# 组合 discriminator reward 和 format reward
if format_reward_tensor is not None:
    format_weight = self.config.trainer.get("format_reward_weight", 0.3)
    reward_tensor = reward_tensor + format_weight * format_reward_tensor
```

### 3. `verl/verl/utils/reward_score/__init__.py` (修改)

注册 GAD 格式奖励到默认的 compute_score 系统：

```python
elif data_source in ["gad_format", "gad", "format_check"]:
    from . import gad_format_reward
    res = gad_format_reward.compute_score(...)
```

### 4. `scripts/train/A3b_gspo/content_merge_trainning/A3b-gad-gspo-optimized.sh` (修改)

训练脚本中启用格式奖励：

```bash
+trainer.use_format_reward=True \
+trainer.format_reward_weight=0.3 \
```

## 使用方法

### 启用格式奖励

在训练脚本中添加以下参数：

```bash
+trainer.use_format_reward=True \
+trainer.format_reward_weight=0.3 \
+trainer.print_format_reward_details=True \
+trainer.print_format_problem_samples=2 \
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `trainer.use_format_reward` | False | 是否启用格式奖励 |
| `trainer.format_reward_weight` | 0.3 | 格式奖励权重 (0-1) |
| `trainer.print_format_reward_details` | True | 是否打印详细的格式奖励统计 |
| `trainer.print_format_problem_samples` | 2 | 打印有问题的样本数量 |
| `trainer.print_sample_interval` | 10 | 打印样本的步数间隔 |
| `trainer.print_sample_num` | 2 | 每次打印的样本数量 |

### TensorBoard 监控指标

启用格式奖励后，以下指标会被记录到 TensorBoard：

#### 格式奖励指标 (format/)
| 指标 | 说明 |
|-----|------|
| `format/reward_avg` | 格式奖励平均值 |
| `format/reward_min` | 格式奖励最小值 |
| `format/reward_max` | 格式奖励最大值 |
| `format/penalty_ratio` | 有惩罚的样本比例 |
| `format/json_missing_ratio` | JSON 缺失比例 |
| `format/json_incomplete_ratio` | JSON 不完整比例 |
| `format/json_invalid_ratio` | JSON 无效比例 |
| `format/thinking_leak_ratio` | 思考泄露比例 |
| `format/mixed_language_ratio` | 中英混杂比例 |
| `format/repetition_ratio` | 重复输出比例 |
| `format/too_long_ratio` | 输出过长比例 |
| `format/too_short_ratio` | 输出过短比例 |

#### Reward 组合指标 (reward/)
| 指标 | 说明 |
|-----|------|
| `reward/discriminator_mean` | Discriminator reward 平均值 |
| `reward/discriminator_min` | Discriminator reward 最小值 |
| `reward/discriminator_max` | Discriminator reward 最大值 |
| `reward/format_contribution_mean` | 格式奖励贡献平均值 |
| `reward/combined_mean` | 组合后 reward 平均值 |
| `reward/format_ratio` | 格式奖励占总 reward 的比例 |

### 控制台输出示例

```
================================================================================
[Step 100] Format Reward Statistics:
================================================================================
  Average Score: -0.1234 (range: [-0.8000, 0.0500])
  Samples with Penalties: 15/128 (11.7%)

  Penalty Breakdown:
    - format/json_missing: 3/128 (2.3%), avg_penalty=0.500
    - language/thinking_leak: 2/128 (1.6%), avg_penalty=0.400
    - content/repetition_consecutive: 5/128 (3.9%), avg_penalty=0.500

  Problem Sample Examples:

    --- Sample 12 (score=-0.400) ---
    Penalties: {'language': {'type': 'thinking_leak', 'penalty': 0.4}}
    Response: Here is the analysis: {"conclusion": "是", "analysis": "...

================================================================================

[Step 100] Reward Combination:
  Discriminator: mean=0.8500, range=[0.2000, 1.5000]
  Format (weight=0.3): mean=-0.0370, range=[-0.2400, 0.0150]
  Combined: mean=0.8130, range=[-0.0400, 1.5150]
  Format Ratio: 4.35%
```

## 完整的优化参数

| 参数 | 原值 | 优化值 | 作用 |
|-----|------|-------|------|
| kl_loss_coef | 0.2 | 0.4 | 防止模式崩溃 |
| clip_ratio_low | 1e-3 | 0.005 | 允许有效学习 |
| clip_ratio_high | 2e-3 | 0.01 | 允许有效学习 |
| actor_lr | 1e-6 | 5e-7 | 更稳定的训练 |
| actor_grad_clip | 0.5 | 0.3 | 更稳定的训练 |
| max_response_length | 128 | 256 | 防止 JSON 截断 |
| temperature | 0.8 | 0.6 | 减少随机性 |
| critic_warmup | 0 | 5 | discriminator 预热 |
| use_format_reward | - | True | 格式惩罚 |
| format_reward_weight | - | 0.3 | 格式奖励权重 |

## 注意事项

1. 格式奖励只在 GAD 模式下生效（`use_critic=True` 且 `use_rm=False`）
2. 格式奖励与 discriminator 奖励相加，权重由 `format_reward_weight` 控制
3. 建议同时调整 prompt 结构，将聊天记录放在中间，输出要求放在最后
4. **新增**：格式奖励以 ground_truth 为参照，只有当 ground_truth 是 JSON 时才检查 JSON 格式


## 格式奖励传递到 Student 模型的完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Discriminator 计算 value                                     │
│     discriminator_reward = batch.batch["values"]                │
│     reward_tensor = discriminator_reward.clone()                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. 格式奖励计算并组合                                            │
│     format_contribution = format_weight * format_reward_tensor  │
│     reward_tensor = reward_tensor + format_contribution         │
│     ← 这里 reward_tensor 已经包含格式奖励                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 设置 token_level_scores                                      │
│     batch.batch["token_level_scores"] = reward_tensor           │
│     ← 组合后的 reward 被存入 batch                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 转换为 token_level_rewards                                   │
│     batch.batch["token_level_rewards"] = token_level_scores     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. GRPO Advantage 计算                                          │
│     scores = token_level_rewards.sum(dim=-1)                    │
│     ← 格式奖励在这里被累加到 scores 中                             │
│     advantages = (scores - mean) / std                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. Actor (Student) 更新                                         │
│     actor_output = self.actor_rollout_wg.update_actor(batch)    │
│     ← batch 中包含 advantages，用于 GSPO policy loss 计算         │
└─────────────────────────────────────────────────────────────────┘
```

**结论**：格式奖励能够传递到 Student 模型的 GSPO 更新中。

关键代码路径：
1. `reward_tensor = discriminator_reward + format_weight * format_reward_tensor`
2. `batch.batch["token_level_scores"] = reward_tensor`
3. `batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]`
4. GRPO: `scores = token_level_rewards.sum(dim=-1)` → 计算 advantages
5. Actor 使用 advantages 计算 policy gradient loss

格式奖励通过影响 `token_level_rewards → advantages → pg_loss` 的链条，最终影响 Student 模型的参数更新。

## 算法代码详解

### JSON 提取与解析

```python
def extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取第一个完整的 JSON 对象"""
    # 使用深度计数和字符串状态追踪
    # 正确处理转义字符和嵌套结构

def parse_json_safe(text: str) -> Tuple[Optional[dict], str]:
    """
    安全解析 JSON，返回 (解析结果, 错误类型)
    错误类型: "ok", "missing", "incomplete", "invalid", "prefix", "empty"
    """
```

### JSON 一致性检测

```python
def check_json_consistency(solution: str, ground_truth: str) -> Tuple[Optional[Dict], Optional[dict]]:
    """
    核心逻辑：
    1. 如果 ground_truth 是有效 JSON，则 solution 也必须是有效 JSON
    2. 如果 ground_truth 包含特定字段，检查 solution 是否也包含
    3. 如果 ground_truth 是纯文本（非 JSON），则不检查 JSON 格式
    """
```

### 语言污染检测

```python
def check_language_pollution(text: str, parsed_json: Optional[dict] = None) -> Optional[Dict]:
    """
    检测三类语言污染：
    1. 英文思考泄露模式：here is, based on, according to, let me, I will 等
    2. 中英文混杂：中文后跟多个英文单词
    3. JSON 值中的英文句子：递归检查所有 JSON 值
    """
```

### n-gram 重复率计算

```python
def compute_ngram_repetition(text: str, ngram_size: int = 4) -> float:
    """
    计算字符级 n-gram 重复率
    返回值范围 [0, 1]，越高表示重复越严重
    """
```

### 内容问题检测

```python
def check_content_issues(text: str, ground_truth: str = "") -> Optional[Dict]:
    """
    检测五类内容问题：
    1. 连续重复：10+ 字符重复 3 次以上
    2. n-gram 重复：重复率 > 35%
    3. 双重输出：JSON 前有大段文本（> 50 字符）
    4. 时间戳泄露：[YYYY-MM-DD HH:MM:SS] 格式
    5. 长度异常：与 ground_truth 比较
    """
```
