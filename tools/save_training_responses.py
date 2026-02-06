"""
在训练过程中保存实际使用的responses

用途：
1. 保存训练时实际使用的student和teacher responses
2. 用于后续验证Critic的评分一致性
3. 分析训练时和测试时responses的差异

使用方法：
在 dp_critic.py 的 update_critic 方法中调用此函数
"""

import torch
import os
from datetime import datetime


def save_training_responses(
    step: int,
    model_inputs: dict,
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    save_dir: str = "./saved_responses",
    save_interval: int = 50,
    num_samples: int = 10
):
    """
    保存训练时的responses和分数
    
    Args:
        step: 当前训练步数
        model_inputs: 包含input_ids, responses, teacher_response等的字典
        teacher_score: Teacher的分数
        student_score: Student的分数
        save_dir: 保存目录
        save_interval: 保存间隔（每N步保存一次）
        num_samples: 每次保存的样本数量
    """
    
    # 只在指定的步数保存
    if step % save_interval != 0:
        return
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备保存的数据
    batch_size = min(num_samples, teacher_score.size(0))
    
    save_data = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        
        # 输入数据
        'input_ids': model_inputs['input_ids'][:batch_size].cpu(),
        'attention_mask': model_inputs['attention_mask'][:batch_size].cpu(),
        'position_ids': model_inputs.get('position_ids', None),
        
        # Responses
        'student_responses': model_inputs['responses'][:batch_size].cpu(),
        'teacher_responses': model_inputs['teacher_response'][:batch_size].cpu(),
        
        # Masks
        'student_response_mask': model_inputs['attention_mask'][:batch_size, -model_inputs['responses'].size(1):].cpu(),
        'teacher_response_mask': model_inputs['teacher_attention_mask'][:batch_size, -model_inputs['teacher_response'].size(1):].cpu(),
        
        # 分数
        'teacher_scores': teacher_score[:batch_size].cpu(),
        'student_scores': student_score[:batch_size].cpu(),
        
        # 统计信息
        'batch_d_acc': (teacher_score > student_score).float().mean().item(),
        'batch_score_diff': (teacher_score - student_score).mean().item(),
    }
    
    # 保存
    save_path = os.path.join(save_dir, f"responses_step_{step}.pt")
    torch.save(save_data, save_path)
    
    print(f"✅ Saved training responses to {save_path}")
    print(f"   Samples: {batch_size}, d_acc: {save_data['batch_d_acc']:.2%}, score_diff: {save_data['batch_score_diff']:.4f}")


# ============================================================================
# 在 dp_critic.py 中的使用示例
# ============================================================================

"""
在 verl/verl/workers/critic/dp_critic.py 的 update_critic 方法中添加：

# 在计算完teacher_score和student_score之后
if use_discriminator:
    # ... 现有的代码 ...
    
    # 计算d_acc
    d_acc = (teacher_score > student_score).float().mean()
    
    # 🔧 添加：保存训练时的responses
    from tools.save_training_responses import save_training_responses
    
    # 每50步保存一次，保存10个样本
    save_training_responses(
        step=self._update_step,
        model_inputs=model_inputs,
        teacher_score=teacher_score,
        student_score=student_score,
        save_dir="/home/jovyan/JQ/gad_gspo_B300/saved_responses",
        save_interval=50,
        num_samples=10
    )
    
    # ... 继续现有的代码 ...
"""


# ============================================================================
# 验证脚本：使用保存的responses测试Critic
# ============================================================================

def test_critic_with_saved_responses(
    critic_model,
    tokenizer,
    saved_responses_path: str
):
    """
    使用保存的responses测试Critic
    
    Args:
        critic_model: Critic模型
        tokenizer: Tokenizer
        saved_responses_path: 保存的responses文件路径
    
    Returns:
        dict: 包含测试结果的字典
    """
    import torch
    
    print("="*80)
    print(f"使用保存的responses测试Critic")
    print(f"文件: {saved_responses_path}")
    print("="*80)
    
    # 加载保存的数据
    data = torch.load(saved_responses_path)
    
    print(f"\n加载的数据:")
    print(f"  Step: {data['step']}")
    print(f"  Timestamp: {data['timestamp']}")
    print(f"  Samples: {data['teacher_scores'].size(0)}")
    print(f"  训练时d_acc: {data['batch_d_acc']:.2%}")
    print(f"  训练时score_diff: {data['batch_score_diff']:.4f}")
    
    # 将数据移到GPU
    device = next(critic_model.pretrained_model.parameters()).device
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    student_responses = data['student_responses'].to(device)
    teacher_responses = data['teacher_responses'].to(device)
    
    # 重新计算分数
    print(f"\n重新计算分数...")
    
    teacher_scores_new = []
    student_scores_new = []
    
    critic_model.eval()
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            # 计算teacher分数
            teacher_input_ids = torch.cat([
                input_ids[i:i+1, :-teacher_responses.size(1)],
                teacher_responses[i:i+1]
            ], dim=1)
            
            teacher_outputs = critic_model(
                input_ids=teacher_input_ids,
                attention_mask=attention_mask[i:i+1],
                use_cache=False
            )
            teacher_values = teacher_outputs[2][:, -teacher_responses.size(1):]
            if teacher_values.dim() == 3:
                teacher_values = teacher_values.squeeze(-1)
            
            # 排除EOS token
            teacher_response_mask = data['teacher_response_mask'][i:i+1].to(device)
            teacher_response_ids = teacher_responses[i:i+1]
            eos_token_id = tokenizer.eos_token_id
            is_eos = (teacher_response_ids == eos_token_id)
            teacher_mask_no_eos = teacher_response_mask & (~is_eos)
            
            teacher_score = (teacher_values * teacher_mask_no_eos).sum() / teacher_mask_no_eos.sum().clamp(min=1)
            teacher_scores_new.append(teacher_score.item())
            
            # 计算student分数（类似的过程）
            student_input_ids = torch.cat([
                input_ids[i:i+1, :-student_responses.size(1)],
                student_responses[i:i+1]
            ], dim=1)
            
            student_outputs = critic_model(
                input_ids=student_input_ids,
                attention_mask=attention_mask[i:i+1],
                use_cache=False
            )
            student_values = student_outputs[2][:, -student_responses.size(1):]
            if student_values.dim() == 3:
                student_values = student_values.squeeze(-1)
            
            student_response_mask = data['student_response_mask'][i:i+1].to(device)
            student_response_ids = student_responses[i:i+1]
            is_eos = (student_response_ids == eos_token_id)
            student_mask_no_eos = student_response_mask & (~is_eos)
            
            student_score = (student_values * student_mask_no_eos).sum() / student_mask_no_eos.sum().clamp(min=1)
            student_scores_new.append(student_score.item())
    
    # 计算新的d_acc
    teacher_scores_new = torch.tensor(teacher_scores_new)
    student_scores_new = torch.tensor(student_scores_new)
    
    d_acc_new = (teacher_scores_new > student_scores_new).float().mean().item()
    score_diff_new = (teacher_scores_new - student_scores_new).mean().item()
    
    # 对比结果
    print(f"\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    print(f"\n训练时记录的分数:")
    print(f"  Teacher平均分: {data['teacher_scores'].mean().item():.4f}")
    print(f"  Student平均分: {data['student_scores'].mean().item():.4f}")
    print(f"  d_acc: {data['batch_d_acc']:.2%}")
    print(f"  score_diff: {data['batch_score_diff']:.4f}")
    
    print(f"\n重新计算的分数:")
    print(f"  Teacher平均分: {teacher_scores_new.mean().item():.4f}")
    print(f"  Student平均分: {student_scores_new.mean().item():.4f}")
    print(f"  d_acc: {d_acc_new:.2%}")
    print(f"  score_diff: {score_diff_new:.4f}")
    
    print(f"\n差异:")
    teacher_diff = abs(data['teacher_scores'].mean().item() - teacher_scores_new.mean().item())
    student_diff = abs(data['student_scores'].mean().item() - student_scores_new.mean().item())
    d_acc_diff = abs(data['batch_d_acc'] - d_acc_new)
    
    print(f"  Teacher分数差异: {teacher_diff:.4f}")
    print(f"  Student分数差异: {student_diff:.4f}")
    print(f"  d_acc差异: {d_acc_diff:.2%}")
    
    # 诊断
    print(f"\n" + "="*80)
    print("诊断")
    print("="*80)
    
    if teacher_diff < 0.1 and student_diff < 0.1 and d_acc_diff < 0.05:
        print("✅ 分数一致性良好！")
        print("   Critic的评分是稳定的")
        print("   问题可能在于测试时使用了不同的responses")
    else:
        print("⚠️  分数不一致！")
        print("   可能的原因:")
        print("   1. Critic模型加载有问题")
        print("   2. 推理代码与训练代码不一致")
        print("   3. 数值精度问题")
    
    return {
        'training_d_acc': data['batch_d_acc'],
        'test_d_acc': d_acc_new,
        'training_teacher_mean': data['teacher_scores'].mean().item(),
        'test_teacher_mean': teacher_scores_new.mean().item(),
        'training_student_mean': data['student_scores'].mean().item(),
        'test_student_mean': student_scores_new.mean().item(),
    }


if __name__ == "__main__":
    # 示例：测试保存的responses
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import AutoModelForCausalLMWithValueHead
    
    critic_path = "/home/jovyan/JQ/gad_gspo_B300/models/opd-v9-1-29-fsdp2/global_step_500/critic_merged"
    saved_responses_path = "/home/jovyan/JQ/gad_gspo_B300/saved_responses/responses_step_300.pt"
    
    # 加载模型
    print("加载Critic模型...")
    tokenizer = AutoTokenizer.from_pretrained(critic_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        critic_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    
    # 测试
    results = test_critic_with_saved_responses(
        critic_model,
        tokenizer,
        saved_responses_path
    )
