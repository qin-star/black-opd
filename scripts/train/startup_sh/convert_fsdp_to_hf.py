#!/usr/bin/env python3
"""
将 FSDP 分片格式的模型转换为 HuggingFace 标准格式
"""

import os
import sys
import torch
import json
import shutil
from pathlib import Path
import argparse
from transformers import AutoConfig, AutoTokenizer

def convert_fsdp_to_hf(fsdp_path, output_path):
    """
    将 FSDP 分片模型转换为 HuggingFace 标准格式
    
    Args:
        fsdp_path: FSDP 分片模型路径
        output_path: 输出的标准格式模型路径
    """
    print(f"🔄 转换 FSDP 模型: {fsdp_path}")
    print(f"📁 输出路径: {output_path}")
    
    # 检查输入路径
    fsdp_path = Path(fsdp_path)
    if not fsdp_path.exists():
        raise FileNotFoundError(f"FSDP 模型路径不存在: {fsdp_path}")
    
    # 查找所有分片文件
    shard_files = list(fsdp_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"未找到 FSDP 分片文件: {fsdp_path}")
    
    print(f"📦 找到 {len(shard_files)} 个分片文件")
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 复制配置文件和分词器文件
    config_files = [
        "config.json",
        "tokenizer_config.json", 
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
        "chat_template.jinja"
    ]
    
    print("📋 复制配置文件...")
    for file_name in config_files:
        src_file = fsdp_path / file_name
        if src_file.exists():
            dst_file = output_path / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  ✓ {file_name}")
    
    # 加载和合并模型权重
    print("🔗 合并模型权重...")
    
    try:
        # 尝试使用 torch.load 加载分片
        merged_state_dict = {}
        
        # 按 rank 顺序加载分片
        shard_files.sort(key=lambda x: int(x.name.split('_rank_')[1].split('.')[0]))
        
        for i, shard_file in enumerate(shard_files):
            print(f"  📦 加载分片 {i+1}/{len(shard_files)}: {shard_file.name}")
            
            try:
                # 加载分片数据
                shard_data = torch.load(shard_file, map_location='cpu')
                
                # 提取模型权重 (可能在不同的键下)
                if isinstance(shard_data, dict):
                    # 查找模型权重
                    model_weights = None
                    for key in ['model_state_dict', 'state_dict', 'model']:
                        if key in shard_data:
                            model_weights = shard_data[key]
                            break
                    
                    if model_weights is None:
                        # 如果没有找到特定键，假设整个字典就是权重
                        model_weights = shard_data
                    
                    # 合并权重
                    for param_name, param_tensor in model_weights.items():
                        if param_name in merged_state_dict:
                            # 如果参数已存在，可能需要拼接
                            print(f"    ⚠️  参数 {param_name} 已存在，跳过")
                        else:
                            merged_state_dict[param_name] = param_tensor
                            
            except Exception as e:
                print(f"    ❌ 加载分片失败: {e}")
                continue
        
        if not merged_state_dict:
            raise ValueError("未能从分片中提取任何模型权重")
        
        print(f"✅ 成功合并 {len(merged_state_dict)} 个参数")
        
        # 保存合并后的模型
        output_model_file = output_path / "pytorch_model.bin"
        print(f"💾 保存模型: {output_model_file}")
        
        torch.save(merged_state_dict, output_model_file)
        
        # 验证保存的模型
        file_size = output_model_file.stat().st_size / (1024**3)  # GB
        print(f"📊 模型文件大小: {file_size:.2f} GB")
        
        if file_size < 1.0:
            print("⚠️  警告: 模型文件大小异常小，可能转换不完整")
        
        print("✅ 转换完成！")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将 FSDP 分片模型转换为 HuggingFace 标准格式")
    parser.add_argument("fsdp_path", help="FSDP 分片模型路径")
    parser.add_argument("output_path", help="输出的标准格式模型路径")
    parser.add_argument("--force", action="store_true", help="强制覆盖输出目录")
    
    args = parser.parse_args()
    
    # 检查输出路径
    if Path(args.output_path).exists() and not args.force:
        response = input(f"输出路径已存在: {args.output_path}\n是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("❌ 操作已取消")
            return
    
    # 执行转换
    success = convert_fsdp_to_hf(args.fsdp_path, args.output_path)
    
    if success:
        print("\n🎉 转换成功完成！")
        print(f"📁 标准格式模型已保存到: {args.output_path}")
        print("\n📋 现在可以使用标准格式模型进行训练:")
        print(f"   STAGE1_MODEL=\"{args.output_path}\"")
    else:
        print("\n❌ 转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
