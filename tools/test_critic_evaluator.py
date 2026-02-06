#!/usr/bin/env python3
"""
测试 Critic 评估器功能

用法：
    python tools/test_critic_evaluator.py \
        --critic_path /path/to/critic/model \
        --actor_path /path/to/actor/model \
        --eval_data /path/to/eval.parquet \
        --num_samples 10
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from verl.trainer.ppo.critic_evaluator import CriticEvaluator


def load_model(model_path: str, model_type: str = 'critic'):
    """加载模型"""
    print(f"Loading {model_type} model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if model_type == 'critic':
        # Critic 需要 value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        model.eval()
    else:
        # Actor 直接使用 base model
        model = base_model
        model.eval()
    
    print(f"✅ {model_type.capitalize()} model loaded")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="测试 Critic 评估器")
    parser.add_argument("--critic_path", type=str, required=True,
                       help="Critic 模型路径")
    parser.add_argument("--actor_path", type=str, required=True,
                       help="Actor 模型路径")
    parser.add_argument("--eval_data", type=str, required=True,
                       help="评估数据集路径")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="评估样本数")
    parser.add_argument("--n_resp_per_prompt", type=int, default=4,
                       help="每个 prompt 生成的 responses 数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批处理大小")
    parser.add_argument("--output_dir", type=str, default="./test_eval_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Critic 评估器测试")
    print("=" * 80)
    print(f"Critic 模型: {args.critic_path}")
    print(f"Actor 模型: {args.actor_path}")
    print(f"评估数据: {args.eval_data}")
    print(f"样本数: {args.num_samples}")
    print(f"每个 prompt 的 responses: {args.n_resp_per_prompt}")
    print()
    
    # 加载模型
    critic_model, tokenizer = load_model(args.critic_path, 'critic')
    actor_model, _ = load_model(args.actor_path, 'actor')
    
    # 创建评估器
    print("Creating evaluator...")
    evaluator = CriticEvaluator(
        config={},
        critic_module=critic_model,
        actor_module=actor_model,
        tokenizer=tokenizer,
        eval_data_path=args.eval_data,
        eval_freq=1,  # 每步都评估（测试用）
        num_eval_samples=args.num_samples,
        n_resp_per_prompt=args.n_resp_per_prompt,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        generation_config={
            'temperature': 0.6,
            'top_p': 0.9,
            'max_new_tokens': 512,
            'do_sample': True,
            'repetition_penalty': 1.2,
        }
    )
    
    # 执行评估
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    metrics = evaluator.evaluate(step=0)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print(f"\n✅ 评估完成！结果已保存到: {args.output_dir}")
    print(f"   - 详细结果: {args.output_dir}/eval_step_0_results.json")
    print(f"   - 历史记录: {args.output_dir}/eval_history.csv")


if __name__ == "__main__":
    main()
