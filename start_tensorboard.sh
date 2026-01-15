#!/bin/bash
# 启动 TensorBoard 脚本

# 设置变量
backend=fsdp2
experiment_name=black-opd-A3b-gspo-warmup-$backend
TENSORBOARD_DIR="/home/jovyan/JQ/gad_gspo_B300/tensorboard/${experiment_name}"
TENSORBOARD_PORT=6019

# 创建目录
mkdir -p ${TENSORBOARD_DIR}

# 检查端口是否被占用
if lsof -Pi :${TENSORBOARD_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "端口 ${TENSORBOARD_PORT} 已被占用，TensorBoard 可能已在运行"
    echo "访问地址: http://localhost:${TENSORBOARD_PORT}"
else
    echo "启动 TensorBoard..."
    echo "日志目录: ${TENSORBOARD_DIR}"
    echo "端口: ${TENSORBOARD_PORT}"
    
    # 启动 TensorBoard
    nohup tensorboard --logdir=${TENSORBOARD_DIR} --port=${TENSORBOARD_PORT} --bind_all > /tmp/tensorboard.log 2>&1 &
    
    sleep 2
    echo "TensorBoard 已启动: http://localhost:${TENSORBOARD_PORT}"
fi