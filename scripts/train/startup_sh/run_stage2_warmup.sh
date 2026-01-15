#!/bin/bash
# ============================================================
# 阶段 2: Warmup (PPO + Critic 预热)
# 训练 Critic（判别器），为对抗训练做准备
# ============================================================

set -e

# ========== 配置 ==========
# 直接使用已转换的模型（跳过自动检测和转换）
STAGE1_MODEL="/home/jovyan2/opd_rl/models/chengla-8b-seqkd/global_step_64/actor_converted"
CRITIC_MODEL="/home/jovyan2/opd_rl/model/Qwen3-8B"
DATA_PATH="/home/jovyan2/opd_rl/data/chengla_train.parquet"
VAL_DATA_PATH="/home/jovyan2/opd_rl/data/chengla_test.parquet"
EXP_NAME="chengla-8b-warmup"
NNODES=1
N_GPUS=4

# 验证已转换的模型
echo "🔍 验证已转换的模型: $STAGE1_MODEL"

if [ ! -d "$STAGE1_MODEL" ]; then
    echo "❌ 模型目录不存在: $STAGE1_MODEL"
    echo "   请先运行转换脚本或检查路径是否正确"
    exit 1
fi

if [ ! -f "$STAGE1_MODEL/pytorch_model.bin" ] && [ ! -f "$STAGE1_MODEL/model.safetensors" ]; then
    echo "❌ 未找到标准格式模型文件"
    echo "   请确认转换是否成功完成"
    exit 1
fi

echo "✅ 找到已转换的标准格式模型"

# GPU 可见性设置（可以修改这里指定使用哪些 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用所有 4 个 GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3        # 只使用前 4 个 GPU
# export CUDA_VISIBLE_DEVICES=0,1            # 只使用前 2 个 GPU

echo "============================================================"
echo "阶段 2: Warmup 训练"
echo "============================================================"
echo "输入模型: $STAGE1_MODEL"
echo "数据路径: $DATA_PATH"
echo "实验名称: $EXP_NAME"
echo "节点数: $NNODES"
echo "============================================================"


# 设置 PyTorch 显存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置环境变量以避免分片策略问题
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0

# 禁用一些可能导致问题的 PyTorch 功能
export TORCH_COMPILE_DISABLE=1
export TORCH_SHOW_CPP_STACKTRACES=1

# 检查 PyTorch 版本并设置兼容性
echo "🔍 检查 PyTorch 版本..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
if hasattr(torch.distributed, 'is_available'):
    print(f'分布式可用: {torch.distributed.is_available()}')
"

# 设置更保守的分布式配置
export TORCH_DISTRIBUTED_DETAIL_DEBUG=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 强制禁用分布式张量和相关功能
export TORCH_DISABLE_DISTRIBUTED_TENSOR=1
export TORCH_DISABLE_FUNCTIONAL_TENSOR=1
export TORCH_DISABLE_DYNAMO=1
export TORCH_DISABLE_AUTOGRAD_FUNCTION_CACHE=1

# 设置更保守的内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# 最终验证模型目录和配置文件
if [ ! -d "$STAGE1_MODEL" ]; then
    echo "❌ 模型目录不存在: $STAGE1_MODEL"
    exit 1
fi

if [ ! -f "$STAGE1_MODEL/config.json" ]; then
    echo "❌ 模型配置文件不存在: $STAGE1_MODEL/config.json"
    exit 1
fi

echo "✅ 模型验证通过"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 切换分支
echo "🔄 切换到 warmup 分支..."
cd verl
git checkout warmup
cd ..

# 备份原始脚本
cp scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh.bak

# 修改数据路径和输出路径
echo "🔧 修改训练脚本的数据路径和输出路径..."
sed -i "s|data.train_files=/tmp/lmsys_gpt5_chat_4k_filtered_train.parquet|data.train_files=$DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh
sed -i "s|data.val_files=/tmp/lmsys_gpt5_chat_4k_filtered_test.parquet|data.val_files=$VAL_DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh
sed -i "s|trainer.default_local_dir=/tmp/\${EXP_NAME}|trainer.default_local_dir=/home/jovyan2/opd_rl/models/\${EXP_NAME}|g" scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh

# 开始训练
echo "🚀 开始 Warmup 训练..."
bash /home/jovyan2/opd_rl/scripts/train/chengla_8B/chengla-warmup.sh \
  --model "$STAGE1_MODEL" \
  --exp_name "$EXP_NAME" \
  --nnodes $NNODES \
  --reward_model "$CRITIC_MODEL"

echo "============================================================"
echo "✅ 阶段 2 (Warmup) 训练完成！"
echo "输出模型: /home/jovyan2/opd_rl/models/$EXP_NAME/actor"
echo "输出 Critic: /home/jovyan2/opd_rl/models/$EXP_NAME/critic"
echo "============================================================"
echo ""
echo "📋 下一步："
echo "bash run_stage3_gad.sh"
echo "============================================================"
