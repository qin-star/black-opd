#!/bin/bash
# ============================================================
# Knowledge Analysis A3b GSPO Training Script (Stage 3: GAD Adversarial Training)
# Based on official GSPO training script format
# ============================================================

set -x

# ===================================== Environment =====================================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ===================================== Model & Experiment =====================================
# 使用 warmup 阶段训练后的模型
# 注意: 需要先运行 merge 命令将 FSDP checkpoint 合并为 HuggingFace 格式:
# python verl/scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /home/jovyan/JQ/gad_gspo_B300/models/core_content-A3b-gspo-gad-fsdp/global_step_12/actor \
#     --target_dir /home/jovyan/JQ/gad_gspo_B300/models/core_content-A3b-gspo-gad-fsdp/global_step_12/actor_merged \
#     --hf_model_path /mnt/public_data/Qwen/Qwen3-30B-A3B-Instruct-2507

# python verl/scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /home/jovyan/JQ/gad_gspo_B300/models/core-content-A3b-gspo-warmup-fsdp/global_step_7/critic \
#     --target_dir /home/jovyan/JQ/gad_gspo_B300/models/core-content-A3b-gspo-warmup-fsdp/global_step_7/critic_merged \
#     --hf_model_path /mnt/public_data/Qwen/Qwen3-8B

actor_model_path="/home/jovyan/JQ/gad_gspo_B300/models/core-content-A3b-gspo-warmup-fsdp/global_step_7/actor_merged"
critic_model_path="/home/jovyan/JQ/gad_gspo_B300/models/core-content-A3b-gspo-warmup-fsdp/global_step_7/critic_merged"

backend=fsdp  # fsdp, fsdp2, megatron
project_name=kowledge_analysis_A3b
experiment_name=core_content-A3b-gspo-gad-new-$backend
default_local_dir=/home/jovyan/JQ/gad_gspo_B300/models/$experiment_name

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo

# Reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.2    # 降低 KL 约束，允许 actor 更多探索
kl_loss_type=low_var_kl  # 更稳定的 KL loss 类型

# GSPO clip ratio: 控制 importance ratio 的范围 [1-low, 1+high]
# 原值 3e-4/4e-4 太小，导致 63% 更新被裁剪
# 标准 PPO 用 0.2，GSPO 建议用较小值但不能太小
clip_ratio_low=0.1
clip_ratio_high=0.1

actor_lr=3e-6     # 提高学习率，加速 actor 学习
actor_grad_clip=0.5  # 与 warmup 一致
critic_lr=1e-6    # 大幅降低 Critic 学习率，让其慢慢学习
critic_grad_clip=0.2  # 与 warmup 一致
gae_gamma=1.0
gae_lam=0.95
# GAD 阶段: 增加 critic_warmup，让判别器先稳定学习
critic_warmup=10

# ===================================== Data/Model =====================================
train_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/core_content/core_content_train.parquet
test_files=/home/jovyan/JQ/gad_gspo/scripts/data_process/trainning_data/konwledge_analysis/analysis_knowledge_val.parquet

max_prompt_length=1550    # 与 warmup 一致
max_response_length=256   # 修改：数据集 P99=174，设置 256 足够
train_batch_size=128      # 修改：减小 batch size，增加训练步数
val_batch_size=128
ppo_mini_batch_size=64    # 相应减小

# ===================================== Training =====================================
# B300 268GB 显存，大幅提升 token 容量
actor_max_token_len_per_gpu=65536
critic_max_token_len_per_gpu=65536

# FSDP parallelism config
USP_SIZE=1

# Actor model config - inline without variables to avoid escaping issues

# Select config based on backend
if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
fi

# ===================================== Inference (vLLM Rollout) =====================================
rollout_name=vllm
infer_tp=2
gpu_memory_utilization=0.70  # 降低推理显存占用，为训练留出更多空间
max_model_len=2048   # 修改：1550 + 256 + 256(余量)
n_resp_per_prompt=8  # 保持：每条生成 8 个 response
temperature=0.8      # 修改：略微降低，减少无意义生成

# ===================================== Ray Isolation (B300) =====================================
# 为 B300 GPU 配置独立的 Ray 环境，避免与其他训练任务冲突
export RAY_TMPDIR=/tmp/ray_b300
export RAY_ADDRESS=""  # 确保不连接到已有的 Ray 集群
mkdir -p ${RAY_TMPDIR}

# 清理本实例的残留 Ray 进程
echo "清理 B300 残留的 Ray 进程..."
RAY_TMPDIR=${RAY_TMPDIR} ray stop --force 2>/dev/null || true
rm -rf ${RAY_TMPDIR}/* 2>/dev/null || true
sleep 2


# ===================================== Environment Variables =====================================
# 使用本地 verl 代码，而非系统安装的版本
export PYTHONPATH="/home/jovyan/JQ/gad_gspo_B300/verl:${PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=36000
export WANDB_INIT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT='chengla_gspo'
export WANDB_API_KEY='YOUR_WANDB_API_KEY'
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

# PyTorch memory optimization
# NOTE: expandable_segments MUST be False for vLLM compatibility
export PYTORCH_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# ===================================== TensorBoard Setup =====================================
export TENSORBOARD_DIR="/home/jovyan/JQ/gad_gspo_B300/tensorboard/${experiment_name}"
mkdir -p ${TENSORBOARD_DIR}

# 日志目录
LOG_DIR="/home/jovyan/JQ/gad_gspo_B300/logs/${experiment_name}"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

TENSORBOARD_PORT=6020

# 检查端口是否被占用
if lsof -Pi :${TENSORBOARD_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "端口 ${TENSORBOARD_PORT} 已被占用，跳过 TensorBoard 启动"
else
    nohup tensorboard --logdir=${TENSORBOARD_DIR} --port=${TENSORBOARD_PORT} --bind_all > /tmp/tensorboard_${TENSORBOARD_PORT}.log 2>&1 &
    TB_PID=$!
    sleep 2
    
    if ps -p $TB_PID > /dev/null 2>&1; then
        echo "========================================="
        echo "   TensorBoard 已启动！"
        echo "   访问: http://localhost:${TENSORBOARD_PORT}"
        echo "   进程 ID: $TB_PID"
        echo "========================================="
    else
        echo "TensorBoard 启动失败，查看日志: /tmp/tensorboard_${TENSORBOARD_PORT}.log"
    fi
fi

echo ""
echo "========================================="
echo "开始 GAD 训练 (Stage 3: Adversarial Training)"
echo "========================================="
echo ""

# ===================================== Launch Training =====================================
# GAD Training Configuration (Stage 3: Adversarial Training)
# If your data contains 'teacher_response' field, GAD discriminator mode will be automatically enabled
# critic_warmup=0: Train both critic and actor from step 0
#
# IMPORTANT: dataloader_num_workers MUST be 0 to avoid pickle serialization issues
# Use ++ to force override existing config value
# Use relative path to config directory (from script location to verl/verl/trainer/config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/../../../../verl/verl/trainer/config"

python3 -m verl.trainer.main_ppo \
    --config-path=${CONFIG_PATH} \
    --config-name=$CONFIG_NAME \
    ++data.dataloader_num_workers=0 \
    ++data.num_proc=1 \
    ++data.filter_overlong_prompts_workers=1 \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    data.prompt_key=content \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.grad_clip=$actor_grad_clip \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE \
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.router_replay.mode="R3" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    critic.enable=True \
    critic.strategy=$backend \
    critic.model.path=$critic_model_path \
    critic.model.use_shm=True \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.optim.lr=$critic_lr \
    critic.grad_clip=$critic_grad_clip \
    critic.ppo_mini_batch_size=$ppo_mini_batch_size \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$USP_SIZE \
    reward_model.enable=False \
    trainer.use_legacy_worker_impl=enable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    +trainer.default_tensorboard_dir=${TENSORBOARD_DIR} \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.save_freq=16 \
    trainer.test_freq=-1 \
    trainer.total_epochs=8 \
    trainer.total_training_steps=null \
    2>&1 | tee ${LOG_FILE}