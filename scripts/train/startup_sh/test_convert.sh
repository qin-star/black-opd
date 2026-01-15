#!/bin/bash
# 测试 FSDP 转换功能

set -e

echo "============================================================"
echo "测试 FSDP 模型转换"
echo "============================================================"

FSDP_PATH="/home/jovyan2/opd_rl/models/chengla-8b-seqkd/global_step_64/actor"
OUTPUT_PATH="/home/jovyan2/opd_rl/models/chengla-8b-seqkd/global_step_64/actor_converted"

echo ""
echo "🔍 检查 FSDP 模型路径..."
if [ ! -d "$FSDP_PATH" ]; then
    echo "❌ FSDP 模型路径不存在: $FSDP_PATH"
    exit 1
fi

echo "✅ FSDP 模型路径存在"
echo ""

echo "🔍 检查 FSDP 分片文件..."
if ! ls "$FSDP_PATH"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
    echo "❌ 未找到 FSDP 分片文件"
    exit 1
fi

echo "✅ 找到 FSDP 分片文件:"
ls -lh "$FSDP_PATH"/model_world_size_*_rank_*.pt
echo ""

echo "🔍 检查转换脚本..."
if [ ! -f "scripts/train/startup_sh/convert_fsdp_to_hf.py" ]; then
    echo "❌ 转换脚本不存在"
    exit 1
fi

echo "✅ 转换脚本存在"
echo ""

echo "🔄 开始转换..."
echo "   源路径: $FSDP_PATH"
echo "   目标路径: $OUTPUT_PATH"
echo ""

python3 scripts/train/startup_sh/convert_fsdp_to_hf.py "$FSDP_PATH" "$OUTPUT_PATH" --force

echo ""
echo "============================================================"
echo "🔍 验证转换结果..."
echo "============================================================"

if [ -f "$OUTPUT_PATH/pytorch_model.bin" ]; then
    echo "✅ 转换成功！"
    echo ""
    echo "📁 输出文件:"
    ls -lh "$OUTPUT_PATH"
    echo ""
    echo "📊 模型文件大小:"
    du -sh "$OUTPUT_PATH"
else
    echo "❌ 转换失败，未找到 pytorch_model.bin"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ 测试完成"
echo "============================================================"
