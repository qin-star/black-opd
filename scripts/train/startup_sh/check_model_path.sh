#!/bin/bash
# ============================================================
# 模型路径检查工具
# 用于诊断和定位训练输出的模型文件
# ============================================================

echo "============================================================"
echo "模型路径检查工具"
echo "============================================================"

STAGE1_BASE="/home/jovyan2/opd_rl/models/chenglai-8b-seqkd"

echo ""
echo "📂 检查目录结构: $STAGE1_BASE"
echo "============================================================"

if [ ! -d "$STAGE1_BASE" ]; then
    echo "❌ 基础目录不存在: $STAGE1_BASE"
    exit 1
fi

# 列出所有子目录
echo ""
echo "📁 子目录列表:"
find "$STAGE1_BASE" -maxdepth 2 -type d | sort

# 查找所有 actor 目录
echo ""
echo "🎭 Actor 模型目录:"
find "$STAGE1_BASE" -type d -name "actor"

# 查找所有 checkpoint 目录
echo ""
echo "💾 Checkpoint 目录:"
find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V

# 检查每个可能的模型路径
echo ""
echo "============================================================"
echo "🔍 检查可能的模型路径"
echo "============================================================"

check_model_path() {
    local path=$1
    echo ""
    echo "检查: $path"
    if [ -d "$path" ]; then
        echo "  ✓ 目录存在"
        
        # 检查关键文件
        local has_model=false
        for file in pytorch_model.bin model.safetensors; do
            if [ -f "$path/$file" ]; then
                echo "  ✓ 找到模型文件: $file"
                local size=$(du -h "$path/$file" | cut -f1)
                echo "    大小: $size"
                has_model=true
            fi
        done
        
        # 检查 FSDP 分片格式
        if ls "$path"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
            echo "  ✓ 找到 FSDP 分片模型文件:"
            local total_size=0
            for shard in "$path"/model_world_size_*_rank_*.pt; do
                if [ -f "$shard" ]; then
                    local shard_name=$(basename "$shard")
                    local size=$(du -h "$shard" | cut -f1)
                    echo "    - $shard_name ($size)"
                fi
            done
            has_model=true
        fi
        
        if [ -f "$path/config.json" ]; then
            echo "  ✓ 找到配置文件: config.json"
        else
            echo "  ✗ 缺少配置文件: config.json"
        fi
        
        if [ -f "$path/tokenizer_config.json" ]; then
            echo "  ✓ 找到分词器配置"
        fi
        
        if [ "$has_model" = true ]; then
            echo "  ✅ 这是一个有效的模型路径！"
            return 0
        else
            echo "  ❌ 缺少模型权重文件"
            return 1
        fi
    else
        echo "  ✗ 目录不存在"
        return 1
    fi
}

# 检查常见路径
check_model_path "$STAGE1_BASE/actor"
check_model_path "$STAGE1_BASE/global_step_64/actor"

# 查找并检查所有 checkpoint
CHECKPOINTS=$(find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V)
if [ -n "$CHECKPOINTS" ]; then
    echo ""
    echo "检查所有 checkpoint 中的 actor 模型:"
    echo "$CHECKPOINTS" | while IFS= read -r checkpoint; do
        if [ -n "$checkpoint" ]; then
            check_model_path "$checkpoint/actor"
        fi
    done
fi

echo ""
echo "============================================================"
echo "💡 建议"
echo "============================================================"

# 找到最佳路径
BEST_PATH=""
if [ -d "$STAGE1_BASE/actor" ] && [ -f "$STAGE1_BASE/actor/config.json" ]; then
    BEST_PATH="$STAGE1_BASE/actor"
else
    LATEST=$(find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V | tail -n 1)
    if [ -n "$LATEST" ] && [ -d "$LATEST/actor" ]; then
        BEST_PATH="$LATEST/actor"
    fi
fi

if [ -n "$BEST_PATH" ]; then
    echo "✅ 推荐使用此路径:"
    echo "   $BEST_PATH"
    echo ""
    echo "修改 run_stage2_warmup.sh 中的 STAGE1_BASE 为:"
    echo "   STAGE1_BASE=\"$(dirname $BEST_PATH)\""
else
    echo "❌ 未找到有效的模型路径"
    echo ""
    echo "可能的原因:"
    echo "1. 阶段1 (SeqKD) 训练未完成或失败"
    echo "2. 模型保存路径配置错误"
    echo "3. 磁盘空间不足导致保存失败"
    echo ""
    echo "建议操作:"
    echo "1. 检查阶段1训练日志"
    echo "2. 重新运行: bash run_stage1_seqkd.sh"
fi

echo "============================================================"
