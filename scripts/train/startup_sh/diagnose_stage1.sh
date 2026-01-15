#!/bin/bash
# ============================================================
# 阶段1训练诊断工具
# 分析训练是否正常完成及可能的问题
# ============================================================

echo "============================================================"
echo "阶段1 (SeqKD) 训练诊断工具"
echo "============================================================"

STAGE1_BASE="/home/jovyan2/opd_rl/models/chenglai-8b-seqkd"
EXPECTED_EPOCHS=4
SAVE_FREQ=50

echo ""
echo "📂 基础信息:"
echo "训练目录: $STAGE1_BASE"
echo "预期训练轮数: $EXPECTED_EPOCHS epochs"
echo "保存频率: 每 $SAVE_FREQ steps"
echo ""

# 检查目录是否存在
if [ ! -d "$STAGE1_BASE" ]; then
    echo "❌ 训练目录不存在: $STAGE1_BASE"
    echo "   阶段1训练可能从未开始"
    exit 1
fi

echo "✅ 训练目录存在"

# 检查训练状态文件
echo ""
echo "🔍 检查训练状态文件:"
if [ -f "$STAGE1_BASE/trainer_state.json" ]; then
    echo "✅ 找到 trainer_state.json"
    
    # 尝试解析训练状态
    if command -v python3 &> /dev/null; then
        echo ""
        echo "📊 训练状态分析:"
        python3 -c "
import json
try:
    with open('$STAGE1_BASE/trainer_state.json', 'r') as f:
        state = json.load(f)
    
    print(f'  当前步数: {state.get(\"global_step\", \"未知\")}')
    print(f'  当前轮数: {state.get(\"epoch\", \"未知\")}')
    print(f'  最佳指标: {state.get(\"best_metric\", \"未知\")}')
    print(f'  训练完成: {\"是\" if state.get(\"epoch\", 0) >= $EXPECTED_EPOCHS else \"否\"}')
    
    # 检查是否有保存的checkpoint
    if 'log_history' in state:
        print(f'  日志条目: {len(state[\"log_history\"])} 条')
except Exception as e:
    print(f'  ❌ 解析失败: {e}')
"
    fi
else
    echo "❌ 未找到 trainer_state.json"
    echo "   训练可能从未开始或异常退出"
fi

# 检查所有checkpoint
echo ""
echo "🔍 检查所有 checkpoint:"
CHECKPOINTS=$(find "$STAGE1_BASE" -type d -name "global_step_*" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ 未找到任何 checkpoint"
    echo "   训练可能在第一个保存点之前就失败了"
else
    echo "✅ 找到 checkpoint:"
    
    CHECKPOINT_COUNT=0
    VALID_CHECKPOINTS=0
    LATEST_STEP=0
    
    echo "$CHECKPOINTS" | while read checkpoint; do
        if [ -n "$checkpoint" ]; then
            CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
            step_num=$(basename "$checkpoint" | sed 's/global_step_//')
            
            echo ""
            echo "  📁 Checkpoint: global_step_$step_num"
            
            # 检查actor目录
            if [ -d "$checkpoint/actor" ]; then
                echo "    ✅ actor/ 目录存在"
                
                # 检查关键文件
                if [ -f "$checkpoint/actor/config.json" ]; then
                    echo "    ✅ config.json 存在"
                else
                    echo "    ❌ config.json 缺失"
                fi
                
                # 检查模型权重文件
                model_file=""
                model_size=""
                if [ -f "$checkpoint/actor/pytorch_model.bin" ]; then
                    model_file="pytorch_model.bin"
                    model_size=$(du -h "$checkpoint/actor/pytorch_model.bin" 2>/dev/null | cut -f1)
                elif [ -f "$checkpoint/actor/model.safetensors" ]; then
                    model_file="model.safetensors"
                    model_size=$(du -h "$checkpoint/actor/model.safetensors" 2>/dev/null | cut -f1)
                fi
                
                if [ -n "$model_file" ]; then
                    echo "    ✅ 模型权重: $model_file ($model_size)"
                    VALID_CHECKPOINTS=$((VALID_CHECKPOINTS + 1))
                    
                    # 检查文件大小是否合理 (应该 > 1GB)
                    size_bytes=$(du -b "$checkpoint/actor/$model_file" 2>/dev/null | cut -f1)
                    if [ -n "$size_bytes" ] && [ "$size_bytes" -gt 1000000000 ]; then
                        echo "    ✅ 文件大小正常 (>1GB)"
                    else
                        echo "    ⚠️  文件大小异常 (<1GB)，可能保存不完整"
                    fi
                else
                    echo "    ❌ 模型权重文件缺失"
                fi
                
                # 检查其他文件
                for file in tokenizer_config.json tokenizer.json special_tokens_map.json; do
                    if [ -f "$checkpoint/actor/$file" ]; then
                        echo "    ✅ $file"
                    else
                        echo "    ⚠️  $file 缺失"
                    fi
                done
                
            else
                echo "    ❌ actor/ 目录不存在"
            fi
            
            # 检查critic和ref目录
            if [ -d "$checkpoint/critic" ]; then
                echo "    ✅ critic/ 目录存在"
            else
                echo "    ⚠️  critic/ 目录缺失"
            fi
            
            if [ -d "$checkpoint/ref" ]; then
                echo "    ✅ ref/ 目录存在"
            else
                echo "    ⚠️  ref/ 目录缺失"
            fi
            
            if [ "$step_num" -gt "$LATEST_STEP" ]; then
                LATEST_STEP=$step_num
            fi
        fi
    done
    
    echo ""
    echo "📊 统计信息:"
    echo "  总 checkpoint 数: $(echo "$CHECKPOINTS" | wc -l)"
    echo "  有效 checkpoint 数: 需要手动计算"
    echo "  最新步数: $LATEST_STEP"
    
    # 计算预期的最终步数
    # 假设每个epoch大约有一定数量的steps
    echo "  预期最终步数: 约 $(($EXPECTED_EPOCHS * 50)) (估算)"
fi

# 检查磁盘空间
echo ""
echo "💾 磁盘空间检查:"
df -h "$STAGE1_BASE" | tail -n 1 | while read filesystem size used avail use_percent mount; do
    echo "  文件系统: $filesystem"
    echo "  总空间: $size"
    echo "  已使用: $used"
    echo "  可用空间: $avail"
    echo "  使用率: $use_percent"
    
    # 检查可用空间是否足够 (至少需要20GB)
    avail_gb=$(echo $avail | sed 's/G//' | sed 's/T/*1000/' | bc 2>/dev/null || echo "0")
    if [ "${avail_gb%.*}" -lt 20 ] 2>/dev/null; then
        echo "  ⚠️  可用空间可能不足 (建议 >20GB)"
    else
        echo "  ✅ 可用空间充足"
    fi
done

# 检查最近的日志
echo ""
echo "📋 可能的问题和建议:"

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ 主要问题: 没有任何 checkpoint 被保存"
    echo ""
    echo "可能原因:"
    echo "1. 训练在第一个保存点 (step 50) 之前就失败了"
    echo "2. 磁盘空间不足"
    echo "3. 权限问题"
    echo "4. 训练脚本配置错误"
    echo ""
    echo "建议操作:"
    echo "1. 检查训练日志: 查看控制台输出或日志文件"
    echo "2. 检查磁盘空间: df -h"
    echo "3. 检查权限: ls -la $STAGE1_BASE"
    echo "4. 重新运行训练: bash run_stage1_seqkd.sh"
else
    echo "⚠️  主要问题: checkpoint 存在但模型权重文件缺失"
    echo ""
    echo "可能原因:"
    echo "1. 训练过程中磁盘空间不足，导致保存失败"
    echo "2. 训练被中断，模型保存不完整"
    echo "3. 文件系统错误或权限问题"
    echo ""
    echo "建议操作:"
    echo "1. 清理磁盘空间，确保有足够空间 (>20GB)"
    echo "2. 删除不完整的checkpoint: rm -rf $STAGE1_BASE"
    echo "3. 重新运行阶段1训练: bash run_stage1_seqkd.sh"
    echo "4. 监控训练过程，确保不被中断"
fi

echo ""
echo "============================================================"
echo "诊断完成"
echo "============================================================"
