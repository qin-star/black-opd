#!/bin/bash
# 检查环境配置是否与官方一致

echo "============================================================"
echo "环境配置检查"
echo "============================================================"

echo ""
echo "🔍 Python 版本:"
python3 --version

echo ""
echo "🔍 关键依赖版本:"
python3 -c "
import torch
import transformers
try:
    import vllm
    vllm_version = vllm.__version__
except:
    vllm_version = 'Not installed'

print(f'PyTorch:      {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'vLLM:         {vllm_version}')
print(f'CUDA:         {torch.version.cuda}')
"

echo ""
echo "🔍 verl 分支信息:"
cd verl
echo "当前分支: $(git branch --show-current)"
echo "最新提交: $(git log -1 --oneline)"
echo "远程状态: $(git status -sb)"
cd ..

echo ""
echo "============================================================"
echo "官方推荐配置:"
echo "============================================================"
echo "Python:       3.10.12"
echo "PyTorch:      2.6.0"
echo "vLLM:         0.8.5"
echo "verl 分支:    warmup (from https://github.com/YTianZHU/verl.git)"
echo "============================================================"
