#!/bin/bash
# 启动 Docker 容器并开放权限

# 停止现有容器（如果在运行）
docker stop verl-opt-2 2>/dev/null || true

# 启动容器，开放所有权限
docker start verl-opt-2

# 或者如果需要重新创建容器，使用以下命令：
# docker run -it --privileged \
#   --name verl-opt-2 \
#   --gpus all \
#   -v /path/to/your/data:/home/jovyan/JQ/gad_gspo_B300 \
#   your-image-name

echo "容器已启动，现在可以进入容器："
echo "docker exec -it verl-opt-2 bash"