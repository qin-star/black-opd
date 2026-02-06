#!/usr/bin/env python3
"""分析 TensorBoard 日志文件

使用方法：
    python analyze_tensorboard.py
    python analyze_tensorboard.py --log-file logs/events.out.tfevents.xxx
    python analyze_tensorboard.py --log-dir logs/
"""

import sys
import os
import argparse
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("错误：需要安装 tensorboard")
    print("运行：pip install tensorboard")
    sys.exit(1)

import statistics


def find_latest_event_file(log_dir):
    """查找最新的事件文件"""
    log_path = Path(log_dir)
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    # 返回最新的文件
    return max(event_files, key=lambda p: p.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(description="分析 TensorBoard 日志文件")
    parser.add_argument("--log-file", type=str, help="指定事件文件路径")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录（默认：logs）")
    parser.add_argument("--recent", type=int, default=50, help="分析最近 N 个数据点（默认：50）")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 确定日志文件
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_event_file(args.log_dir)
        if log_file is None:
            print(f"错误：在 {args.log_dir} 中未找到事件文件")
            sys.exit(1)
        print(f"使用最新的日志文件：{log_file}")
    
    if not os.path.exists(log_file):
        print(f"错误：文件不存在 {log_file}")
        sys.exit(1)

    print('\n正在解析 TensorBoard 日志...')
    ea = event_accumulator.EventAccumulator(str(log_file))
    ea.Reload()

    # 列出所有可用的标量指标
    print('\n可用的标量指标：')
    scalar_tags = ea.Tags()['scalars']
    for tag in sorted(scalar_tags):
        print(f'  - {tag}')

    print(f'\n总共 {len(scalar_tags)} 个指标')

    # 提取关键指标
    key_metrics = [
        'critic/d_acc',
        'critic/score_diff',
        'critic/raw_score_diff',
        'critic/teacher_value_mean',
        'critic/student_value_mean',
        'critic/ranking_loss',
        'critic/score_reg',
        'critic/diff_penalty',
        'format/reward_avg',
        'actor/policy_loss',
        'actor/ppo_kl',
    ]

    print('\n' + '='*70)
    print(f'关键指标分析（最近 {args.recent} 个数据点）')
    print('='*70)

    metrics_data = {}
    for metric_name in key_metrics:
        if metric_name in scalar_tags:
            events = ea.Scalars(metric_name)
            if events:
                # 取最近 N 个点
                recent = events[-args.recent:] if len(events) > args.recent else events
                values = [e.value for e in recent]
                
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                min_val = min(values)
                max_val = max(values)
                latest_val = values[-1]
                
                metrics_data[metric_name] = {
                    'latest': latest_val,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'count': len(events)
                }
                
                print(f'\n{metric_name}:')
                print(f'  最新值: {latest_val:.4f}')
                print(f'  平均值: {mean_val:.4f} ± {std_val:.4f}')
                print(f'  范围: [{min_val:.4f}, {max_val:.4f}]')
                print(f'  数据点数: {len(events)}')
        else:
            print(f'\n{metric_name}: 未找到')

    # 诊断分析
    print('\n' + '='*70)
    print('诊断分析')
    print('='*70)

    # 计算健康度评分
    total_score = 0
    max_score = 100

    # 1. score_diff 评分（40分）
    if 'critic/score_diff' in metrics_data:
        score_diff = metrics_data['critic/score_diff']['mean']
        print(f'\n[1. score_diff 分析]')
        print(f'  当前值: {score_diff:.4f}')
        
        if score_diff < 0.1:
            score_diff_points = 40
            status = "优秀 - Student 已接近 Teacher"
        elif score_diff < 0.5:
            score_diff_points = 30
            status = "良好 - Student 质量不错"
        elif score_diff < 1.0:
            score_diff_points = 20
            status = "可接受 - Student 还在学习"
        elif score_diff < 2.0:
            score_diff_points = 10
            status = "需要关注 - 差距较大"
        else:
            score_diff_points = 0
            status = "有问题 - 差距过大"
        
        total_score += score_diff_points
        print(f'  状态: {status}')
        print(f'  得分: {score_diff_points}/40')

    # 2. ranking_loss 评分（30分）
    if 'critic/ranking_loss' in metrics_data:
        ranking_loss = metrics_data['critic/ranking_loss']['mean']
        print(f'\n[2. ranking_loss 分析]')
        print(f'  当前值: {ranking_loss:.4f}')
        print(f'  理论最优值: log(2) ≈ 0.693')
        
        if 0.65 <= ranking_loss <= 0.72:
            ranking_loss_points = 30
            status = "优秀 - 接近理论最优"
        elif (0.50 <= ranking_loss < 0.65) or (0.72 < ranking_loss <= 0.80):
            ranking_loss_points = 20
            status = "良好 - 在合理范围"
        elif 0.30 <= ranking_loss < 0.50:
            ranking_loss_points = 10
            status = "需要关注 - 区分太容易"
        else:
            ranking_loss_points = 0
            status = "有问题 - 异常"
        
        total_score += ranking_loss_points
        print(f'  状态: {status}')
        print(f'  得分: {ranking_loss_points}/30')

    # 3. 绝对分数评分（20分）
    if 'critic/teacher_value_mean' in metrics_data and 'critic/student_value_mean' in metrics_data:
        teacher_value = metrics_data['critic/teacher_value_mean']['mean']
        student_value = metrics_data['critic/student_value_mean']['mean']
        print(f'\n[3. 绝对分数分析]')
        print(f'  Teacher 平均分: {teacher_value:.4f}')
        print(f'  Student 平均分: {student_value:.4f}')
        
        if abs(teacher_value) <= 2 and abs(student_value) <= 2:
            abs_score_points = 20
            status = "优秀 - 无数值漂移"
        elif abs(teacher_value) <= 5 and abs(student_value) <= 5:
            abs_score_points = 10
            status = "可接受 - 轻微漂移"
        else:
            abs_score_points = 0
            status = "有问题 - 数值漂移"
        
        total_score += abs_score_points
        print(f'  状态: {status}')
        print(f'  得分: {abs_score_points}/20')

    # 4. d_acc 评分（10分）
    if 'critic/d_acc' in metrics_data:
        d_acc = metrics_data['critic/d_acc']['mean']
        print(f'\n[4. d_acc 分析]')
        print(f'  当前值: {d_acc:.4f}')
        
        if d_acc < 0.70:
            d_acc_points = 10
            status = "理想状态"
        elif d_acc < 0.85:
            d_acc_points = 8
            status = "良好状态"
        elif d_acc < 0.95:
            d_acc_points = 5
            status = "可接受"
        else:
            d_acc_points = 0
            status = "需要结合其他指标判断"
        
        total_score += d_acc_points
        print(f'  状态: {status}')
        print(f'  得分: {d_acc_points}/10')

    # 总分和建议
    print('\n' + '='*70)
    print('综合评估')
    print('='*70)
    print(f'\n总分: {total_score}/{max_score}')
    
    if total_score >= 80:
        health = "优秀"
        recommendation = "训练非常健康，继续当前配置"
    elif total_score >= 60:
        health = "良好"
        recommendation = "训练基本健康，可以继续"
    elif total_score >= 40:
        health = "需要关注"
        recommendation = "观察更多步数，可能需要调整"
    elif total_score >= 20:
        health = "需要调整"
        recommendation = "存在明显问题，建议调整参数"
    else:
        health = "需要修复"
        recommendation = "训练有严重问题，必须修改"
    
    print(f'健康度: {health}')
    print(f'建议: {recommendation}')
    
    # 具体建议
    print('\n详细建议:')
    
    if 'critic/score_diff' in metrics_data and 'critic/d_acc' in metrics_data:
        score_diff = metrics_data['critic/score_diff']['mean']
        d_acc = metrics_data['critic/d_acc']['mean']
        
        if score_diff < 0.1 and d_acc > 0.95:
            print('  - 这是正常的训练状态！')
            print('  - score_diff 很小说明 Student 已接近 Teacher')
            print('  - d_acc 高是因为累积效应，不是问题')
            print('  - 建议：继续当前训练，不需要修改代码')
        elif score_diff > 1.0 and d_acc > 0.95:
            print('  - 质量差距较大，可能需要调整')
            print('  - 建议：检查 format/reward_avg 是否过低')
            print('  - 如果需要，考虑增大 temperature 到 3.0')
    
    if 'critic/ranking_loss' in metrics_data:
        ranking_loss = metrics_data['critic/ranking_loss']['mean']
        if ranking_loss < 0.3:
            print('  - ranking_loss 过低，建议增大 temperature')
        elif 0.65 <= ranking_loss <= 0.72:
            print('  - ranking_loss 接近理论最优，训练收敛良好')
    
    if 'critic/teacher_value_mean' in metrics_data and 'critic/student_value_mean' in metrics_data:
        teacher_value = metrics_data['critic/teacher_value_mean']['mean']
        student_value = metrics_data['critic/student_value_mean']['mean']
        if abs(teacher_value) > 5 or abs(student_value) > 5:
            print('  - 检测到数值漂移，建议增大 score_reg 到 0.02')
    
    print('\n' + '='*70)
    print('分析完成！详细指南请参考：GAD训练指标分析指南.md')
    print('='*70 + '\n')


if __name__ == "__main__":
    main()
