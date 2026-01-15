#!/bin/bash
# ============================================================
# 阶段 3: GAD (Generative Adversarial Distillation)
# 完整的对抗蒸馏训练
# ============================================================

set -e

# ========== 配置 ==========
STAGE2_ACTOR="/home/jovyan2/opd_rl/models/chenglai-8b-warmup/actor"
STAGE2_CRITIC="/home/jovyan2/opd_rl/models/chenglai-8b-warmup/critic"
DATA_PATH="/home/jovyan2/opd_rl/data/chengla_train.parquet"
VAL_DATA_PATH="/home/jovyan2/opd_rl/data/chengla_test.parquet"
EXP_NAME="chenglai-8b-gad"
NNODES=1
N_GPUS=8

# GPU 可见性设置（可以修改这里指定使用哪些 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用所有 8 个 GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3        # 只使用前 4 个 GPU
# export CUDA_VISIBLE_DEVICES=0,1            # 只使用前 2 个 GPU

echo "============================================================"
echo "阶段 3: GAD 训练"
echo "============================================================"
echo "输入 Actor: $STAGE2_ACTOR"
echo "输入 Critic: $STAGE2_CRITIC"
echo "数据路径: $DATA_PATH"
echo "实验名称: $EXP_NAME"
echo "节点数: $NNODES"
echo "============================================================"

# 检查文件
if [ ! -d "$STAGE2_ACTOR" ]; then
    echo "❌ 阶段2 Actor 不存在: $STAGE2_ACTOR"
    echo "请先运行: bash run_stage2_warmup.sh"
    exit 1
fi

if [ ! -d "$STAGE2_CRITIC" ]; then
    echo "❌ 阶段2 Critic 不存在: $STAGE2_CRITIC"
    echo "请先运行: bash run_stage2_warmup.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 切换分支
echo "🔄 切换到 gad 分支..."
cd verl
git checkout gad
cd ..

# 备份原始脚本
cp scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh.bak

# 修改数据路径和输出路径
echo "🔧 修改训练脚本的数据路径和输出路径..."
sed -i "s|data.train_files=/tmp/lmsys_gpt5_chat_4k_filtered_train.parquet|data.train_files=$DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh
sed -i "s|data.val_files=/tmp/lmsys_gpt5_chat_4k_filtered_test.parquet|data.val_files=$VAL_DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh
sed -i "s|trainer.default_local_dir=/tmp/\${EXP_NAME}|trainer.default_local_dir=/home/jovyan2/opd_rl/models/\${EXP_NAME}|g" scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh

# 开始训练
echo "🚀 开始 GAD 训练..."
bash scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh \
  --model "$STAGE2_ACTOR" \
  --critic "$STAGE2_CRITIC" \
  --exp_name "$EXP_NAME" \
  --nnodes $NNODES

echo "============================================================"
echo "✅ 阶段 3 (GAD) 训练完成！"
echo "最终模型: /home/jovyan2/opd_rl/models/$EXP_NAME/actor"
echo "============================================================"
echo ""
echo "🔄 转换为 HuggingFace 格式："
echo "python -m verl.utils.hf_ckpt_io \\"
echo "  --load_dir /home/jovyan2/opd_rl/models/$EXP_NAME/actor \\"
echo "  --save_dir /home/jovyan2/opd_rl/models/chenglai-8b-final \\"
echo "  --save_type hf"
echo "============================================================"
