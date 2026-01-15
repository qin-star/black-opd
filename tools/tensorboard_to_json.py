#!/usr/bin/env python3
"""
TensorBoard日志转JSON工具
将TensorBoard事件文件转换为JSON格式，方便LLM分析
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import tensorflow as tf
except ImportError:
    print("请安装tensorboard: pip install tensorboard tensorflow")
    exit(1)


def extract_tensorboard_data(log_dir: str) -> Dict[str, Any]:
    """
    从TensorBoard日志目录提取数据
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 获取所有标量数据
    scalar_tags = event_acc.Tags()['scalars']
    
    data = {
        'metadata': {
            'log_dir': log_dir,
            'extraction_time': datetime.now().isoformat(),
            'scalar_tags': scalar_tags
        },
        'scalars': {},
        'summary': {}
    }
    
    # 提取每个标量的数据
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        
        values = []
        for event in scalar_events:
            values.append({
                'step': event.step,
                'wall_time': event.wall_time,
                'value': float(event.value)
            })
        
        data['scalars'][tag] = values
        
        # 添加统计摘要
        if values:
            data['summary'][tag] = {
                'total_steps': len(values),
                'min_value': min(v['value'] for v in values),
                'max_value': max(v['value'] for v in values),
                'final_value': values[-1]['value'],
                'first_step': values[0]['step'],
                'last_step': values[-1]['step']
            }
    
    return data


def process_tensorboard_logs(base_dir: str, output_file: str = None):
    """
    处理TensorBoard日志目录，支持嵌套结构
    """
    base_path = Path(base_dir)
    all_data = {}
    
    # 查找所有包含事件文件的目录
    for root, dirs, files in os.walk(base_path):
        # 检查是否包含TensorBoard事件文件
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        
        if event_files:
            relative_path = os.path.relpath(root, base_path)
            print(f"处理日志目录: {relative_path}")
            
            try:
                data = extract_tensorboard_data(root)
                all_data[relative_path] = data
            except Exception as e:
                print(f"处理 {relative_path} 时出错: {e}")
                continue
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tensorboard_data_{timestamp}.json"
    
    # 保存为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"数据已保存到: {output_file}")
    
    # 打印摘要信息
    print("\n=== 数据摘要 ===")
    for log_path, data in all_data.items():
        print(f"\n日志路径: {log_path}")
        print(f"标量指标数量: {len(data['scalars'])}")
        
        for tag, summary in data['summary'].items():
            print(f"  - {tag}: {summary['total_steps']} 步, "
                  f"最终值: {summary['final_value']:.4f}")
    
    return all_data


def generate_analysis_prompt(data: Dict[str, Any], output_file: str = None):
    """
    生成用于LLM分析的提示文本
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_prompt_{timestamp}.txt"
    
    prompt = "# 训练数据分析请求\n\n"
    prompt += "请分析以下训练数据，重点关注：\n"
    prompt += "1. 训练收敛情况\n"
    prompt += "2. 损失函数变化趋势\n"
    prompt += "3. 性能指标表现\n"
    prompt += "4. 可能的训练问题或建议\n\n"
    
    prompt += "## 训练数据概览\n\n"
    
    for log_path, log_data in data.items():
        prompt += f"### 实验: {log_path}\n\n"
        
        for tag, summary in log_data['summary'].items():
            prompt += f"**{tag}**:\n"
            prompt += f"- 训练步数: {summary['total_steps']}\n"
            prompt += f"- 数值范围: {summary['min_value']:.4f} ~ {summary['max_value']:.4f}\n"
            prompt += f"- 最终值: {summary['final_value']:.4f}\n"
            prompt += f"- 步数范围: {summary['first_step']} ~ {summary['last_step']}\n\n"
    
    prompt += "\n## 详细数据\n\n"
    prompt += "```json\n"
    prompt += json.dumps(data, indent=2, ensure_ascii=False)
    prompt += "\n```\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"LLM分析提示已保存到: {output_file}")
    return prompt


def main():
    parser = argparse.ArgumentParser(description='将TensorBoard日志转换为JSON格式')
    parser.add_argument('--log_dir', '-d', default='tensorboard_log', 
                       help='TensorBoard日志目录路径')
    parser.add_argument('--output', '-o', help='输出JSON文件路径')
    parser.add_argument('--generate_prompt', '-p', action='store_true',
                       help='生成LLM分析提示文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"错误: 日志目录 {args.log_dir} 不存在")
        return
    
    # 转换数据
    data = process_tensorboard_logs(args.log_dir, args.output)
    
    # 生成分析提示 
    if args.generate_prompt:
        generate_analysis_prompt(data)


if __name__ == "__main__":
    main()