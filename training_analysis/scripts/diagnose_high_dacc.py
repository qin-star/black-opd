#!/usr/bin/env python3
"""
诊断 d_acc 持续高位的问题

这个脚本帮助你快速定位 d_acc 高的根本原因。

使用方法：
    python tools/diagnose_high_dacc.py --log-dir logs/your_training_run

输出：
    - 问题诊断报告
    - 推荐的修复方案
    - 预期效果
"""

import argparse
import json
from pathlib import Path


def parse_tensorboard_logs(log_dir):
    """解析 TensorBoard 日志"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("错误：需要安装 tensorboard")
        print("运行：pip install tensorboard")
        return None
    
    metrics = {
        "critic/d_acc": [],
        "critic/score_diff": [],
        "critic/raw_score_diff": [],
        "critic/teacher_value_mean": [],
        "critic/student_value_mean": [],
        "critic/ranking_loss": [],
        "critic/score_reg": [],
        "critic/diff_penalty": [],
        "actor/format_reward_mean": [],
    }
    
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"警告：在 {log_dir} 中未找到 TensorBoard 事件文件")
        return None
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        for metric_name in metrics.keys():
            if metric_name in ea.Tags()["scalars"]:
                events = ea.Scalars(metric_name)
                for event in events:
                    metrics[metric_name].append({
                        "step": event.step,
                        "value": event.value,
                    })
    
    return metrics


def diagnose_high_dacc(metrics):
    """诊断 d_acc 高的原因"""
    if not metrics or not metrics.get("critic/d_acc"):
        return {"error": "无法获取指标数据"}
    
    import statistics
    
    # 提取最近的数据（最后 50 个点）
    def get_recent_stats(metric_list, n=50):
        if not metric_list:
            return None
        recent = metric_list[-n:] if len(metric_list) > n else metric_list
        values = [m["value"] for m in recent]
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }
    
    diagnosis = {}
    
    # 1. d_acc 统计
    d_acc_stats = get_recent_stats(metrics["critic/d_acc"])
    diagnosis["d_acc"] = d_acc_stats
    
    # 2. score_diff 统计
    score_diff_stats = get_recent_stats(metrics["critic/score_diff"])
    diagnosis["score_diff"] = score_diff_stats
    
    # 3. teacher/student value 统计
    teacher_value_stats = get_recent_stats(metrics["critic/teacher_value_mean"])
    student_value_stats = get_recent_stats(metrics["critic/student_value_mean"])
    diagnosis["teacher_value"] = teacher_value_stats
    diagnosis["student_value"] = student_value_stats
    
    # 4. loss 组件统计
    ranking_loss_stats = get_recent_stats(metrics["critic/ranking_loss"])
    score_reg_stats = get_recent_stats(metrics["critic/score_reg"])
    diff_penalty_stats = get_recent_stats(metrics["critic/diff_penalty"])
    diagnosis["ranking_loss"] = ranking_loss_stats
    diagnosis["score_reg"] = score_reg_stats
    diagnosis["diff_penalty"] = diff_penalty_stats
    
    # 5. actor 训练效果
    format_reward_stats = get_recent_stats(metrics.get("actor/format_reward_mean", []))
    diagnosis["format_reward"] = format_reward_stats
    
    return diagnosis


def print_diagnosis_report(diagnosis):
    """打印诊断报告"""
    print("\n" + "="*70)
    print("d_acc 持续高位问题诊断报告")
    print("="*70)
    
    if "error" in diagnosis:
        print(f"\n错误：{diagnosis['error']}")
        return
    
    # 1. d_acc 分析
    print("\n【1. Discriminator 准确率分析】")
    d_acc = diagnosis["d_acc"]
    print(f"  当前值：{d_acc['latest']:.4f}")
    print(f"  平均值：{d_acc['mean']:.4f} ± {d_acc['std']:.4f}")
    print(f"  范围：[{d_acc['min']:.4f}, {d_acc['max']:.4f}]")
    
    if d_acc['mean'] > 0.95:
        print("  ⚠️  问题：d_acc 过高（>95%），存在以下可能：")
        print("      1. Teacher 和 Student 质量差距过大")
        print("      2. Temperature 参数太小")
        print("      3. Loss 设计问题")
    elif d_acc['mean'] > 0.85:
        print("  ⚠️  注意：d_acc 偏高（85-95%），可能需要调整")
    else:
        print("  ✓ d_acc 在合理范围内（<85%）")
    
    # 2. 分数差异分析
    print("\n【2. 分数差异分析】")
    score_diff = diagnosis["score_diff"]
    if score_diff:
        print(f"  当前值：{score_diff['latest']:.4f}")
        print(f"  平均值：{score_diff['mean']:.4f} ± {score_diff['std']:.4f}")
        
        if score_diff['mean'] > 2.0:
            print("  ⚠️  问题：分数差距过大（>2.0）")
            print("      → Teacher 质量远超 Student")
            print("      → 建议：增大 temperature 或降低 format reward 惩罚")
        elif score_diff['mean'] > 1.0:
            print("  ⚠️  注意：分数差距较大（1.0-2.0）")
            print("      → 可能需要调整 temperature")
        else:
            print("  ✓ 分数差距合理（<1.0）")
    
    # 3. 绝对分数分析
    print("\n【3. 绝对分数分析】")
    teacher_value = diagnosis["teacher_value"]
    student_value = diagnosis["student_value"]
    if teacher_value and student_value:
        print(f"  Teacher 平均分：{teacher_value['mean']:.4f}")
        print(f"  Student 平均分：{student_value['mean']:.4f}")
        print(f"  分数比值：{teacher_value['mean'] / (student_value['mean'] + 1e-8):.2f}")
        
        if abs(teacher_value['mean']) > 5 or abs(student_value['mean']) > 5:
            print("  ⚠️  问题：分数绝对值过大")
            print("      → 可能发生数值漂移")
            print("      → 建议：增大 score_reg 权重")
    
    # 4. Loss 组件分析
    print("\n【4. Loss 组件分析】")
    ranking_loss = diagnosis["ranking_loss"]
    score_reg = diagnosis["score_reg"]
    diff_penalty = diagnosis["diff_penalty"]
    
    if ranking_loss:
        print(f"  Ranking Loss：{ranking_loss['mean']:.4f}")
        if ranking_loss['mean'] < 0.3:
            print("      ⚠️  过低：loss 对差异不够敏感")
            print("      → 建议：减小 temperature")
        elif ranking_loss['mean'] > 0.7:
            print("      ⚠️  过高：loss 对差异过于敏感")
            print("      → 建议：增大 temperature")
    
    if score_reg:
        print(f"  Score Reg：{score_reg['mean']:.4f}")
        if score_reg['mean'] < 0.01:
            print("      ⚠️  过低：正则化不足")
            print("      → 建议：增大 score_reg 权重")
    
    if diff_penalty:
        print(f"  Diff Penalty：{diff_penalty['mean']:.4f}")
        if diff_penalty['mean'] > 0.5:
            print("      ⚠️  过高：频繁触发过度自信惩罚")
            print("      → 建议：降低阈值或增大 temperature")
    
    # 5. Actor 训练效果
    print("\n【5. Actor 训练效果】")
    format_reward = diagnosis["format_reward"]
    if format_reward:
        print(f"  Format Reward：{format_reward['mean']:.4f}")
        if format_reward['mean'] < -0.5:
            print("  ⚠️  问题：Student 质量很差")
            print("      → Format reward 惩罚过重")
            print("      → 建议：降低 format reward 惩罚力度")
        elif format_reward['mean'] < 0:
            print("  ⚠️  注意：Student 质量有待提升")
        else:
            print("  ✓ Student 质量良好")
    
    # 6. 综合诊断
    print("\n【6. 综合诊断与建议】")
    print("\n根据以上分析，推荐的修复方案：")
    
    recommendations = []
    
    # 根据不同情况给出建议
    if d_acc['mean'] > 0.95:
        if score_diff and score_diff['mean'] > 2.0:
            recommendations.append({
                "priority": "高",
                "action": "增大 temperature",
                "detail": "将 temperature 从 2.0 增大到 5.0 或 10.0",
                "file": "verl/verl/trainer/ppo/core_algos.py",
                "line": "~1467",
                "expected": "d_acc 应该下降到 70-85%"
            })
        
        if diff_penalty and diff_penalty['mean'] > 0.3:
            recommendations.append({
                "priority": "中",
                "action": "降低 diff_penalty 阈值",
                "detail": "将阈值从 1.5 降低到 0.5",
                "file": "verl/verl/trainer/ppo/core_algos.py",
                "line": "~1475",
                "expected": "限制 critic 放大差距"
            })
        
        if score_reg and score_reg['mean'] < 0.01:
            recommendations.append({
                "priority": "中",
                "action": "增大 score_reg 权重",
                "detail": "将权重从 0.005 增大到 0.02",
                "file": "verl/verl/trainer/ppo/core_algos.py",
                "line": "~1472",
                "expected": "防止分数漂移"
            })
        
        if format_reward and format_reward['mean'] < -0.5:
            recommendations.append({
                "priority": "低",
                "action": "降低 format reward 惩罚",
                "detail": "减小各项惩罚的系数（例如乘以 0.5）",
                "file": "verl/verl/utils/reward_score/gad_format_reward.py",
                "expected": "Student 质量提升"
            })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. [{rec['priority']}优先级] {rec['action']}")
            print(f"     修改：{rec['detail']}")
            print(f"     文件：{rec['file']} (约第 {rec['line']} 行)")
            print(f"     预期：{rec['expected']}")
    else:
        print("\n  ✓ 未检测到明显问题，继续观察训练")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="诊断 d_acc 持续高位的问题")
    parser.add_argument("--log-dir", type=str, required=True, help="TensorBoard 日志目录")
    
    args = parser.parse_args()
    
    print("正在解析训练日志...")
    metrics = parse_tensorboard_logs(args.log_dir)
    
    if metrics is None:
        return 1
    
    print("正在分析指标...")
    diagnosis = diagnose_high_dacc(metrics)
    
    print_diagnosis_report(diagnosis)
    
    return 0


if __name__ == "__main__":
    exit(main())
