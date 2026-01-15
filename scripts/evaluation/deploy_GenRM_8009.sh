#!/bin/bash

# === 配置区 ===
MODEL_PATH="/mnt/public_data/nv-community/Qwen3-Nemotron-235B-A22B-GenRM"  # 合并后的完整模型路径
MODEL_NAME="GenRM"                                     # 对外暴露的模型名称（OpenAI API 中使用）
PORT=8009 # 对应透传端口：193
HOST="0.0.0.0"
API_KEY="sk-xxxx"

# GPU 设置
CUDA_VISIBLE_DEVICES=4,5,6,7
GPU_MEMORY_UTILIZATION=0.95    # H100 显存大，可设高些（0.9~0.95）
TENSOR_PARALLEL_SIZE=4         # Qwen3-A3B 若多卡可改为 2/4/8

# 性能调优参数
MAX_MODEL_LEN=32768            # Qwen3 支持长上下文，按需调整（8192 / 16384 / 32768）
MAX_NUM_SEQS=20               # 最大并发序列数（影响吞吐）
ENABLE_CHUNKED_PREFILL=true    # 启用分块预填充，提升长文本吞吐
MAX_NUM_BATCHED_TOKENS=32768   # 动态批处理最大 token 数（越大吞吐越高，但延迟略增）

# === 启动命令 ===
echo "🚀 正在启动 vLLM 服务：Qwen3-8B (gad 版)"
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --dtype bfloat16 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --swap-space 4 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --api-key $API_KEY \
    --uvicorn-log-level info