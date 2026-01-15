#!/bin/bash
# ============================================================
# Chengla 8B GSPO Training Script (Stage 2: Warm-up)
# Based on official GSPO training script format
# ============================================================

set -x

# ===================================== Environment =====================================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ===================================== Model & Experiment =====================================
actor_model_path="/mnt/public_data/Qwen/Qwen3-30B-A3B-Instruct-2507"
critic_model_path="/mnt/public_data/Qwen/Qwen3-8B"  # 使用较小的模型作为 Critic

backend=fsdp2  # fsdp, fsdp2, megatron
project_name=kowledge_analysis_A3b
experiment_name=content-conflict-understanding-A3b-gspo-warmup-$backend
default_local_dir=/home/jovyan/JQ/gad_gspo_B300/models/$experiment_name

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo     # 使用 GSPO loss

# Reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.5    # 大幅增加 KL 约束，KL 爆炸是崩溃的直接原因
kl_loss_type=low_var_kl  # 官方使用的 KL loss 类型，更稳定

clip_ratio_low=3e-4   # GSPO 官方值，sequence-level ratio 需要小的 clip
clip_ratio_high=4e-4  # GSPO 官方值

actor_lr=1e-6     # 增加学习率，让 Actor 能够学习
actor_grad_clip=0.5  # 放宽梯度裁剪
critic_lr=3e-6    # 降低 Critic 学习率，防止 Discriminator 过拟合
critic_grad_clip=0.2  # 官方有梯度裁剪
gae_gamma=1.0
gae_lam=0.95
critic_warmup=20  # 减少 warmup，防止 Discriminator 过拟合

# ===================================== Data/Model =====================================
train_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/content_conflict_understanding_merged/merged_train_test.parquet
test_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/content_conflict_understanding_merged/merged_train_test.parquet

max_prompt_length=550    # core_content 数据集，与 GAD 一致
max_response_length=128   # core_content P99=174，256 足够
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
infer_tp=4 # 4张卡协同处理同意序列不同分
gpu_memory_utilization=0.7
max_model_len=1024   # 550 + 128 + 128(余量)，与 GAD 一致
n_resp_per_prompt=8  # 增加到 8，与官方一致
temperature=0.8  # 增加多样性，让 GRPO 能区分不同 response，防止 Discriminator 过拟合

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

# ===================================== TensorBoard Setup ==========
export TENSORBOARD_DIR="/home/jovyan/JQ/gad_gspo_B300/tensorboard/${experiment_name}"
mkdir -p ${TENSORBOARD_DIR}

# 日志目录
LOG_DIR="/home/jovyan/JQ/gad_gspo_B300/logs/${experiment_name}"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

TENSORBOARD_PORT=6019

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

# ===================================== Permission Check =====================================
WORK_DIR="/home/jovyan/JQ/gad_gspo_B300"

if [ ! -w "$WORK_DIR" ]; then
    echo "[ERROR] Work directory not writable. Fix with: chown -R jovyan:jovyan $WORK_DIR"
    exit 1
fi

mkdir -p /tmp/ray_b300 2>/dev/null
echo "[OK] Permission check passed"

# ===================================== Launch Training =====================================
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
    data.filter_overlong_prompts=True \
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
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE \
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.router_replay.mode="R2" \
    critic.enable=True \
    critic.strategy=$backend \
    critic.model.path=$critic_model_path \
    critic.model.use_shm=True \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.optim.lr=$critic_lr \
    critic.grad_clip=$critic_grad_clip \
    critic.ppo_mini_batch_size=$ppo_mini_batch_size \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$USP_SIZE \
    reward_model.enable=False \
    trainer.use_legacy_worker_impl=enable \
    trainer.critic_warmup=$critic_warmup \
    trainer.warmup_use_sft=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    +trainer.default_tensorboard_dir=${TENSORBOARD_DIR} \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=null \
    2>&1 | tee ${LOG_FILE}