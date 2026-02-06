from tensorboard.backend.event_processing import event_accumulator
import numpy as np

ea = event_accumulator.EventAccumulator('logs/events.out.tfevents.1769084353.ubuntu.2680727.0')
ea.Reload()

# 获取关键指标
score_diff = ea.Scalars('critic/score_diff')
raw_score_diff = ea.Scalars('critic/raw_score_diff')
ranking_loss = ea.Scalars('critic/ranking_loss')
format_reward = ea.Scalars('format/reward_avg')
teacher_score = ea.Scalars('critic/teacher_score_mean')
student_score = ea.Scalars('critic/student_score_mean')

# 分析趋势
def analyze_trend(data, name):
    values = [x.value for x in data]
    steps = [x.step for x in data]
    
    # 分段分析
    early = values[:len(values)//3]
    mid = values[len(values)//3:2*len(values)//3]
    late = values[2*len(values)//3:]
    
    print(f'\n{name}:')
    print(f'  早期 (0-{len(early)}步): {np.mean(early):.4f} ± {np.std(early):.4f}')
    print(f'  中期 ({len(early)}-{len(early)+len(mid)}步): {np.mean(mid):.4f} ± {np.std(mid):.4f}')
    print(f'  后期 ({len(early)+len(mid)}-{len(values)}步): {np.mean(late):.4f} ± {np.std(late):.4f}')
    print(f'  总体变化: {np.mean(late) - np.mean(early):.4f}')
    print(f'  变化率: {(np.mean(late) - np.mean(early)) / abs(np.mean(early)) * 100:.2f}%')

print('='*70)
print('Critic 学习能力分析')
print('='*70)

analyze_trend(score_diff, 'score_diff (归一化质量差异)')
analyze_trend(raw_score_diff, 'raw_score_diff (原始质量差异)')
analyze_trend(ranking_loss, 'ranking_loss (Critic训练损失)')
analyze_trend(format_reward, 'format_reward (Student质量)')
analyze_trend(teacher_score, 'teacher_score (Teacher绝对分数)')
analyze_trend(student_score, 'student_score (Student绝对分数)')

# 计算相关性
format_values = [x.value for x in format_reward]
score_diff_values = [x.value for x in score_diff]
correlation = np.corrcoef(format_values, score_diff_values)[0,1]

print(f'\n相关性分析:')
print(f'  format_reward vs score_diff 相关系数: {correlation:.4f}')
if correlation < 0:
    print(f'  解释: 负相关 - Student质量提升时，质量差异应该缩小 ✓')
else:
    print(f'  解释: 正相关 - Student质量提升时，质量差异不应该扩大 ✗')
