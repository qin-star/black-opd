#!/bin/bash
# 测试模型检测逻辑

echo "============================================================"
echo "测试模型检测逻辑"
echo "============================================================"

STAGE1_BASE="/home/jovyan2/opd_rl/models/chengla-8b-seqkd"

# 函数：检查模型是否有效（包含权重文件）
check_model_valid() {
    local path=$1
    echo "  检查路径: $path"
    if [ -d "$path" ] && [ -f "$path/config.json" ]; then
        echo "    ✓ 目录和config.json存在"
        # 检查标准格式
        if [ -f "$path/pytorch_model.bin" ] || [ -f "$path/model.safetensors" ]; then
            echo "    ✓ 找到标准格式模型文件"
            return 0
        fi
        # 检查 FSDP 分片格式
        if ls "$path"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
            echo "    ✓ 找到 FSDP 分片格式模型"
            return 0
        fi
        echo "    ✗ 未找到模型权重文件"
    else
        echo "    ✗ 目录不存在或缺少config.json"
    fi
    return 1
}

echo ""
echo "🔍 测试模型检测..."

# 测试1: 检查直接actor目录
echo ""
echo "测试1: $STAGE1_BASE/actor"
if check_model_valid "$STAGE1_BASE/actor"; then
    echo "  ✅ 有效"
else
    echo "  ❌ 无效"
fi

# 测试2: 查找checkpoint
echo ""
echo "测试2: 查找所有checkpoint"
if [ -d "$STAGE1_BASE" ]; then
    echo "  基础目录存在: $STAGE1_BASE"
    
    CHECKPOINTS=$(find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V -r)
    if [ -n "$CHECKPOINTS" ]; then
        echo "  找到checkpoint:"
        echo "$CHECKPOINTS"
        
        echo ""
        echo "  测试每个checkpoint的actor目录:"
        for checkpoint in $CHECKPOINTS; do
            echo ""
            echo "  检查: $checkpoint/actor"
            if check_model_valid "$checkpoint/actor"; then
                echo "    ✅ 这个checkpoint有效！"
                LATEST_VALID="$checkpoint/actor"
                break
            else
                echo "    ❌ 这个checkpoint无效"
            fi
        done
        
        if [ -n "$LATEST_VALID" ]; then
            echo ""
            echo "🎉 找到最新有效checkpoint: $LATEST_VALID"
        else
            echo ""
            echo "❌ 没有找到有效的checkpoint"
        fi
    else
        echo "  ❌ 没有找到任何checkpoint"
    fi
else
    echo "  ❌ 基础目录不存在: $STAGE1_BASE"
fi

# 测试3: 检查基础模型
echo ""
echo "测试3: 检查基础模型"
BASE_MODEL="/home/jovyan2/opd_rl/model/Qwen3-8B"
echo "  检查: $BASE_MODEL"
if [ -d "$BASE_MODEL" ] && [ -f "$BASE_MODEL/config.json" ]; then
    echo "    ✅ 基础模型可用"
else
    echo "    ❌ 基础模型不可用"
    echo "    目录存在: $([ -d "$BASE_MODEL" ] && echo "是" || echo "否")"
    echo "    config.json存在: $([ -f "$BASE_MODEL/config.json" ] && echo "是" || echo "否")"
fi

echo ""
echo "============================================================"
echo "测试完成"
echo "============================================================"
