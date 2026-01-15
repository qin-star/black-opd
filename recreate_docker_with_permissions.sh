#!/bin/bash
# 重新创建 Docker 容器并开放权限

echo "========================================="
echo "重新创建 Docker 容器并开放权限"
echo "========================================="

# 获取现有容器的信息
CONTAINER_NAME="verl-opt-2"

echo "1. 获取现有容器信息..."
if docker inspect $CONTAINER_NAME >/dev/null 2>&1; then
    # 获取镜像名
    IMAGE_NAME=$(docker inspect --format='{{.Config.Image}}' $CONTAINER_NAME)
    echo "   镜像名: $IMAGE_NAME"
    
    # 获取挂载信息
    echo "   挂载信息:"
    docker inspect --format='{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' $CONTAINER_NAME
    
    # 获取环境变量
    echo "   环境变量:"
    docker inspect --format='{{range .Config.Env}}{{.}} {{end}}' $CONTAINER_NAME | tr ' ' '\n' | head -10
else
    echo "   容器 $CONTAINER_NAME 不存在"
    exit 1
fi

echo ""
echo "2. 停止并删除现有容器..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo ""
echo "3. 创建新容器（带权限）..."

# 基本的创建命令模板
cat << 'EOF'
请根据你的实际情况修改并运行以下命令：

docker run -it -d \
  --name verl-opt-2 \
  --privileged \
  --gpus all \
  --shm-size=32g \
  -v /path/to/your/host/data:/home/jovyan/JQ/gad_gspo_B300 \
  -v /tmp:/tmp \
  -p 6019:6019 \
  -p 8888:8888 \
  your-image-name

或者如果你想要更精确的权限控制：

docker run -it -d \
  --name verl-opt-2 \
  --gpus all \
  --shm-size=32g \
  --user root \
  --cap-add=SYS_ADMIN \
  -v /path/to/your/host/data:/home/jovyan/JQ/gad_gspo_B300 \
  -v /tmp:/tmp \
  -p 6019:6019 \
  -p 8888:8888 \
  your-image-name

EOF

echo ""
echo "4. 进入容器："
echo "docker exec -it verl-opt-2 bash"