#!/bin/bash
# ============================================================
# Knowledge Analysis A3b GSPO Training Script (Stage 3: GAD Adversarial Training)
# OPTIMIZED VERSION - 解决重复输出、时间戳泄漏、JSON不完整问题
# ============================================================

set -x

# ===================================== Environment =====================================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ===================================== Model & Experiment =====================================
# 使用 warmup 阶段训练后的模型
actor_model_path="/home/jovyan/JQ/gad_gspo_B300/models/subject-1-3-warmup-fsdp2/global_step_375/actor_merged"
critic_model_path="/home/jovyan/JQ/gad_gspo_B300/models/subject-1-3-warmup-fsdp2/global_step_375/critic_merged"

backend=fsdp2
project_name=kowledge_analysis_A3b
experiment_name=subject-1-3-gad-$backend
default_local_dir=/home/jovyan/JQ/gad_gspo_B300/models/$experiment_name

# ===================================== Algorithm (核心优化) =====================================
adv_estimator=grpo
loss_mode=gspo

# Reference policy
use_kl_in_reward=False
kl_coef=0.001

# ========== 优化点 1: KL 约束 ==========
# 问题：从 warmup 的 0.5 骤降到 0.2，导致模型失去约束
# 方案：保持与 warmup 一致的 KL 约束强度
use_kl_loss=True
kl_loss_coef=0.6          # 从 0.2 提高到 0.4，接近 warmup 的 0.5
kl_loss_type=low_var_kl

# ========== 优化点 8: 熵正则化 ==========
# 问题：熵值从 1.42 降到 1.18，模型多样性丧失，导致重复生成
# 方案：添加熵正则化，鼓励模型保持输出多样性
entropy_coeff=0.02        # 从 0.01 提高到 0.02，增强多样性

# ========== 优化点 2: GSPO clip ratio ==========
# 问题：1e-3/2e-3 太小，大量更新被裁剪，模型无法有效学习
# 方案：适度增大，但保持 GSPO 的 sequence-level 特性
clip_ratio_low=0.005      # 从 1e-3 提高到 0.5%
clip_ratio_high=0.01      # 从 2e-3 提高到 1%

# ========== 优化点 3: 学习率 ==========
# 问题：actor_lr=1e-6 可能过高，导致训练不稳定
# 方案：降低 actor 学习率，保持 critic 学习率
actor_lr=3e-6             # 
actor_grad_clip=0.3       # 从 0.5 收紧到 0.3
critic_lr=3e-6            # 保持不变
critic_grad_clip=0.2

gae_gamma=1.0
gae_lam=0.95

# ========== 优化点 4: Critic warmup ==========
# 问题：critic_warmup=0 导致 discriminator 和 actor 同时训练，不稳定
# 方案：给 discriminator 几步预热时间
critic_warmup=20          

# ========== 优化点 7: 格式奖励 ==========
# 启用格式奖励，惩罚时间戳泄漏、重复输出、JSON 不完整
# 触发条件：use_format_reward=True + use_critic=True + reward_model.enable=False
use_format_reward=True
format_reward_weight=0.7  # 格式奖励权重

# ===================================== Data/Model (核心优化) =====================================
train_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/merged/merge-1231.parquet
test_files=/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/merged/merged_train_new_test.parquet


# ========== 优化点 5: Response 长度 ==========
# 问题：128 太短，导致 JSON 被截断
# 方案：增加到 256，确保完整输出
max_prompt_length=3800
max_response_length=2900   # 从 128 提高到 256

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
gpu_memory_utilization=0.75

# ========== 优化点 6: 推理参数 ==========
max_model_len=7200        # 增加以适应更长的 response
n_resp_per_prompt=8
temperature=0.6           # 从 0.8 降低到 0.6，减少随机性

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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=36000
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

TENSORBOARD_PORT=6025

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
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
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
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=null \
    +trainer.format_reward_weight=${format_reward_weight} \
    +trainer.print_format_reward_details=True \
    +trainer.print_format_problem_samples=2 \
    +trainer.print_sample_interval=5 \
    +trainer.print_sample_num=2 \
    2>&1 | tee ${LOG_FILE}