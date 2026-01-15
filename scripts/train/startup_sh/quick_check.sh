#!/bin/bash
# 快速检查模型文件

STAGE1_BASE="/home/jovyan2/opd_rl/models/chenglai-8b-seqkd"

echo "🔍 快速检查模型文件..."
echo "基础目录: $STAGE1_BASE"
echo ""

# 检查目录是否存在
if [ ! -d "$STAGE1_BASE" ]; then
    echo "❌ 目录不存在: $STAGE1_BASE"
    exit 1
fi

# 列出所有 checkpoint
echo "📁 所有 checkpoint:"
find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V

echo ""
echo "🔍 检查每个 checkpoint 的模型文件:"
find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V | while read checkpoint; do
    echo ""
    echo "Checkpoint: $checkpoint"
    if [ -d "$checkpoint/actor" ]; then
        echo "  ✓ actor 目录存在"
        
        # 检查模型文件
        if [ -f "$checkpoint/actor/pytorch_model.bin" ]; then
            size=$(du -h "$checkpoint/actor/pytorch_model.bin" | cut -f1)
            echo "  ✓ pytorch_model.bin ($size)"
        elif [ -f "$checkpoint/actor/model.safetensors" ]; then
            size=$(du -h "$checkpoint/actor/model.safetensors" | cut -f1)
            echo "  ✓ model.safetensors ($size)"
        else
            echo "  ❌ 缺少模型权重文件"
        fi
        
        # 检查配置文件
        if [ -f "$checkpoint/actor/config.json" ]; then
            echo "  ✓ config.json"
        else
            echo "  ❌ 缺少 config.json"
        fi
    else
        echo "  ❌ actor 目录不存在"
    fi
done

echo ""
echo "🔍 检查训练日志和状态文件:"
if [ -f "$STAGE1_BASE/trainer_state.json" ]; then
    echo "  ✓ 找到 trainer_state.json"
fi

if [ -f "$STAGE1_BASE/training_args.bin" ]; then
    echo "  ✓ 找到 training_args.bin"
fi

# 查找最新的有效模型
echo ""
echo "💡 寻找最新的有效模型:"
VALID_MODEL=""
find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V -r | while read checkpoint; do
    if [ -d "$checkpoint/actor" ] && [ -f "$checkpoint/actor/config.json" ]; then
        if [ -f "$checkpoint/actor/pytorch_model.bin" ] || [ -f "$checkpoint/actor/model.safetensors" ]; then
            echo "  ✅ 有效模型: $checkpoint/actor"
            exit 0
        fi
    fi
done

echo ""
echo "📋 建议操作:"
echo "1. 如果没有找到有效模型，需要重新运行阶段1训练"
echo "2. 检查磁盘空间是否充足"
echo "3. 检查阶段1训练日志是否有错误"
