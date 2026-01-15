#!/bin/bash
# ============================================================
# 阶段 1: SeqKD (Sequence Knowledge Distillation)
# 纯 SFT 训练，让 student 模型学习 teacher 的输出
# ============================================================

set -e

# ========== 配置 ==========
MODEL_PATH="/home/jovyan2/opd_rl/model/Qwen3-8B"
DATA_PATH="/home/jovyan2/opd_rl/data/chengla_train.parquet"
VAL_DATA_PATH="/home/jovyan2/opd_rl/data/chengla_test.parquet"
EXP_NAME="chenglai-8b-seqkd"
NNODES=1
N_GPUS=4

# GPU 可见性设置（可以修改这里指定使用哪些 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用所有 4 个 GPU


echo "============================================================"
echo "阶段 1: SeqKD 训练"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "实验名称: $EXP_NAME"
echo "节点数: $NNODES"
echo "============================================================"

# 检查文件
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 切换分支
echo "🔄 切换到 seqkd 分支..."
cd verl
git checkout seqkd
cd ..

# 备份原始脚本
cp scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh.bak

# 修改数据路径和输出路径
echo "🔧 修改训练脚本的数据路径和输出路径..."
sed -i "s|data.train_files=/tmp/lmsys_gpt5_chat_4k_filtered_train.parquet|data.train_files=$DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh
sed -i "s|data.val_files=/tmp/lmsys_gpt5_chat_4k_filtered_test.parquet|data.val_files=$VAL_DATA_PATH|g" scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh
sed -i "s|trainer.default_local_dir=/tmp/\${EXP_NAME}|trainer.default_local_dir=/home/jovyan2/opd_rl/models/\${EXP_NAME}|g" scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh

# 开始训练
echo "🚀 开始 SeqKD 训练..."
bash /home/jovyan2/opd_rl/scripts/train/chengla_8B/chengla-seqkd.sh \
  --model "$MODEL_PATH" \
  --exp_name "$EXP_NAME" \
  --nnodes $NNODES

echo "============================================================"
echo "✅ 阶段 1 (SeqKD) 训练完成！"
echo "输出模型: /home/jovyan2/opd_rl/models/$EXP_NAME/actor"
echo "============================================================"
echo ""
echo "📋 下一步："
echo "bash run_stage2_warmup.sh"
echo "============================================================"
