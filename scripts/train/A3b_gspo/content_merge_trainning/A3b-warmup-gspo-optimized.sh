#!/bin/bash
# ============================================================
# Chengla A3b GSPO Training Script (Stage 2: Warm-up)
# OPTIMIZED VERSION - 与 GAD 阶段参数平滑过渡
# ============================================================

set -x

# ===================================== Environment =====================================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ===================================== Model & Experiment =====================================
actor_model_path="/home/jovyan/work/lns/Task_SFT/merged_model/sft12_ep2/"
critic_model_path="/mnt/public_data/Qwen/Qwen3-8B"

backend=fsdp2
project_name=kowledge_analysis_A3b
experiment_name=content-conflict-understanding-A3b-gspo-warmup-optimized-$backend
default_local_dir=/home/jovyan/JQ/gad_gspo_B300/models/$experiment_name

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo

use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.4          # 与 GAD 阶段保持一致
kl_loss_type=low_var_kl

# GSPO clip ratio - 与 GAD 阶段保持一致
clip_ratio_low=0.005      # 与 GAD 一致
clip_ratio_high=0.01      # 与 GAD 一致

actor_lr=0.0              # 冻结 actor 参数更新
actor_grad_clip=0.3       # 与 GAD 一致
critic_lr=1e-5            # 优化：从 3e-6 提高到 1e-5，加速学习
critic_grad_clip=1.0      # 优化：从 0.2 提高到 1.0，允许更大梯度
gae_gamma=1.0
gae_lam=0.95
critic_warmup=999999      # 设置为极大值，整个训练过程不更新 actor

# ===================================== Data/Model =====================================
train_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/content_conflict_understanding_merged/merged_train_train.parquet
test_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/content_conflict_understanding_merged/merged_train_test.parquet

max_prompt_length=3750
max_response_length=256   # 与 GAD 一致
train_batch_size=128
val_batch_size=128
ppo_mini_batch_size=64

# ===================================== Training =====================================
actor_max_token_len_per_gpu=98304
critic_max_token_len_per_gpu=98304
USP_SIZE=1

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
else
    CONFIG_NAME=ppo_trainer
fi

# ===================================== Inference (vLLM Rollout) =====================================
rollout_name=vllm
infer_tp=4
gpu_memory_utilization=0.7
max_model_len=4500        # 与 GAD 一致
n_resp_per_prompt=8
temperature=0.6           # 与 GAD 一致

# ===================================== Ray Isolation =====================================
export RAY_TMPDIR=/tmp/ray_b300
export RAY_ADDRESS=""
mkdir -p ${RAY_TMPDIR}

echo "清理 B300 残留的 Ray 进程..."
RAY_TMPDIR=${RAY_TMPDIR} ray stop --force 2>/dev/null || true
rm -rf ${RAY_TMPDIR}/* 2>/dev/null || true
sleep 2

# ===================================== Environment Variables =====================================
export PYTHONPATH="/home/jovyan/JQ/gad_gspo_B300/verl:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 修改为实际可用的 GPU 数量
export NCCL_TIMEOUT=36000
export WANDB_INIT_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT='chengla_gspo'
export WANDB_API_KEY='YOUR_WANDB_API_KEY'
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# ===================================== TensorBoard Setup =====================================
export TENSORBOARD_DIR="/home/jovyan/JQ/gad_gspo_B300/tensorboard/${experiment_name}"
mkdir -p ${TENSORBOARD_DIR}

LOG_DIR="/home/jovyan/JQ/gad_gspo_B300/logs/${experiment_name}"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

TENSORBOARD_PORT=6022

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
    trainer.warmup_use_sft=False \
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
