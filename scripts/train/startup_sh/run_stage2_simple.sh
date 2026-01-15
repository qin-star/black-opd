#!/bin/bash
# ============================================================
# 阶段 2: Warmup 训练 (简化版本)
# 使用最保守的配置避免分布式张量问题
# ============================================================

set -e

echo "============================================================"
echo "阶段 2: Warmup 训练 (简化版本)"
echo "============================================================"

# 基本配置
STAGE1_MODEL="/home/jovyan2/opd_rl/models/chengla-8b-seqkd/global_step_64/actor_converted"
CRITIC_MODEL="/home/jovyan2/opd_rl/model/Qwen3-8B"
DATA_PATH="/home/jovyan2/opd_rl/data/chengla_train.parquet"
VAL_DATA_PATH="/home/jovyan2/opd_rl/data/chengla_test.parquet"
EXP_NAME="chengla-8b-warmup-simple"

# 使用单GPU避免分布式问题
export CUDA_VISIBLE_DEVICES=0

# 强制禁用所有可能导致问题的功能
export TORCH_COMPILE_DISABLE=1
export TORCH_DISABLE_DISTRIBUTED_TENSOR=1
export TORCH_DISABLE_FUNCTIONAL_TENSOR=1
export TORCH_DISABLE_DYNAMO=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True

echo "🔍 使用简化配置:"
echo "  模型: $STAGE1_MODEL"
echo "  GPU: 单卡 (GPU 0)"
echo "  批次大小: 4"
echo ""

# 验证模型
if [ ! -d "$STAGE1_MODEL" ]; then
    echo "❌ 模型不存在，使用基础模型"
    STAGE1_MODEL="$CRITIC_MODEL"
fi

# 切换分支
cd verl
git checkout warmup
cd ..

# 运行简化训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=content \
    data.train_files=$DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation=right \
    actor_rollout_ref.model.path=$STAGE1_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=2 \
    critic.model.path=$CRITIC_MODEL \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=False \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.val_before_train=True \
    trainer.critic_warmup=5 \
    trainer.logger=['console'] \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.default_local_dir=/home/jovyan2/opd_rl/models/$EXP_NAME

echo "============================================================"
echo "✅ 简化版本训练完成"
echo "============================================================"
