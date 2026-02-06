"""
测试Critic模型 - 完全模拟训练时的batch结构（独立脚本）

关键改进：
1. 使用train模式（而非eval模式）
2. 使用与训练完全相同的batch结构（teachers和students在同一个batch中）
3. 模拟训练时的n_resp_per_prompt=4配置
"""

import os
import sys

# 🔧 关键修复：在导入torch之前设置CUDA_VISIBLE_DEVICES
# 这必须在任何导入torch的代码之前执行
if '--gpu_ids' in sys.argv:
    gpu_ids_idx = sys.argv.index('--gpu_ids') + 1
    if gpu_ids_idx < len(sys.argv):
        gpu_ids = sys.argv[gpu_ids_idx]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"⚡ 设置CUDA_VISIBLE_DEVICES={gpu_ids}")
elif '--use_multi_gpu' in sys.argv or len(sys.argv) == 1:
    # 默认使用GPU 4,5
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    print(f"⚡ 默认设置CUDA_VISIBLE_DEVICES=4,5")

import requests
import pandas as pd
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import time


# ==================== 配置区域 ====================

# 模型配置
CRITIC_CONFIG = {
    "model_path": "/home/jovyan/JQ/gad_gspo_B300/models/2-3-warmup-v10-fsdp2/global_step_310/critic_merged",
    "use_multi_gpu": True,  # 启用多GPU
}

STUDENT_CONFIG = {
    "url": "http://10.72.1.39:8009/v1/chat/completions",
    "api_key": "sk-xxxx",
    "model_name": "SFT56",
    "temperature": 0.6,  # 与训练时一致
    "repetition_penalty": 1.2
}

# 数据配置
DATA_PATH = "/home/jovyan/JQ/gad_gspo_B300/data/trainning_dataset/subject_1-29/merged/merge-1-29.parquet"

# ================================================


def call_student_api(prompt: str, config: dict) -> str:
    """调用Student API生成response"""
    try:
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": config["model_name"],
            "messages": messages,
            "max_tokens": 512,
            "temperature": config.get("temperature", 0.6),
            "top_p": 0.9,
            "repetition_penalty": config.get("repetition_penalty", 1.2),
        }
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(config["url"], json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        raw_content = result['choices'][0]['message']['content']
        cleaned_content = raw_content.strip()
        
        return cleaned_content
    except Exception as e:
        print(f"❌ Student API调用失败: {e}")
        return ""


def load_critic_model(config: dict):
    """加载Critic模型（使用train模式，支持多GPU）"""
    model_path = config["model_path"]
    use_multi_gpu = config.get("use_multi_gpu", False)
    
    # 注意：CUDA_VISIBLE_DEVICES已经在脚本开头设置
    # 现在GPU 4,5会被映射为cuda:0和cuda:1
    device = "cuda:0"
    
    print(f"🔄 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"🔄 加载模型...")
    
    if use_multi_gpu:
        # 使用多GPU - device_map会自动使用所有可见的GPU
        print(f"⚡ 使用多GPU加载...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"  # auto会自动使用CUDA_VISIBLE_DEVICES中的所有GPU
        )
    else:
        # 单GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device
        )
    
    critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    
    # 🔧 关键：使用train模式（与训练时一致）
    critic_model.train()
    print(f"⚠️  使用train模式（与训练时一致）")
    
    if use_multi_gpu:
        # 打印模型分配情况
        if hasattr(critic_model.pretrained_model, 'hf_device_map'):
            print(f"✅ 模型分配情况:")
            device_map = critic_model.pretrained_model.hf_device_map
            # 统计每个设备上的层数
            device_counts = {}
            for layer, dev in device_map.items():
                dev_str = str(dev)
                device_counts[dev_str] = device_counts.get(dev_str, 0) + 1
            for dev, count in sorted(device_counts.items()):
                print(f"   {dev}: {count} 层")
        else:
            print(f"✅ 模型已分配到多个GPU")
    else:
        print(f"✅ 模型加载到 {device}")
    
    return critic_model, tokenizer


def get_critic_scores_batch(critic_model, tokenizer, prompts: list, responses: list, max_length: int = 2048) -> tuple:
    """
    获取Critic分数（批量推理，与训练代码一致）
    
    Args:
        critic_model: Critic模型
        tokenizer: Tokenizer
        prompts: prompt列表
        responses: response列表
        max_length: 最大序列长度
    
    Returns:
        (scores, lengths): 分数列表和长度列表
    """
    try:
        # 获取设备（支持多GPU）
        if hasattr(critic_model.pretrained_model, 'hf_device_map'):
            # 多GPU模式，使用第一个设备
            device = list(critic_model.pretrained_model.hf_device_map.values())[0]
        else:
            device = next(critic_model.pretrained_model.parameters()).device
        
        batch_size = len(prompts)
        
        # 准备batch数据
        all_input_texts = []
        
        for prompt, response in zip(prompts, responses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_input_texts.append(input_text)
        
        # Batch tokenization（使用padding）
        inputs = tokenizer(
            all_input_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # 🔧 关键修复：计算每个样本的实际response长度
        # 必须在tokenization之后计算，因为chat_template可能添加特殊token
        all_response_lengths = []
        for prompt, response in zip(prompts, responses):
            # 分别tokenize prompt和完整对话
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
            prompt_length = len(prompt_tokens['input_ids'])
            
            # 完整对话的长度
            full_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_tokens = tokenizer(full_text, add_special_tokens=False)
            full_length = len(full_tokens['input_ids'])
            
            # response长度 = 完整长度 - prompt长度
            response_length = full_length - prompt_length
            all_response_lengths.append(response_length)
        
        # 只将inputs移到第一个设备（模型会自动处理多GPU）
        if not hasattr(critic_model.pretrained_model, 'hf_device_map'):
            inputs = inputs.to(device)
        else:
            # 多GPU模式，移到第一个设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Batch前向传播
        with torch.no_grad():
            outputs = critic_model(**inputs, use_cache=False)
            
            # 🔧 关键修复：正确提取values
            # 对于 AutoModelForCausalLMWithValueHead，values在 output[2]
            if hasattr(critic_model, "v_head"):
                # output[2] shape: (batch, seq_len) 或 (batch, seq_len, 1)
                all_values = outputs[2]
                if all_values.dim() == 3:
                    all_values = all_values.squeeze(-1)  # (batch, seq_len)
            else:
                # 对于其他模型类型
                all_values = outputs.logits
                if all_values.dim() == 3:
                    all_values = all_values.squeeze(-1)
            
            # 提取每个样本的values
            scores = []
            lengths = []
            
            # 🔧 调试：打印原始values的统计信息
            if batch_size <= 10:  # 只在小batch时打印
                print(f"\n🔍 调试信息 - 原始values统计:")
                print(f"   all_values shape: {all_values.shape}")
                print(f"   all_values mean: {all_values.mean().item():.4f}")
                print(f"   all_values std: {all_values.std().item():.4f}")
                print(f"   all_values range: [{all_values.min().item():.4f}, {all_values.max().item():.4f}]")
                
                # 打印每个样本的response长度和实际序列长度
                for i in range(min(3, batch_size)):
                    actual_length = inputs['attention_mask'][i].sum().item()
                    response_len = all_response_lengths[i]
                    print(f"   样本{i}: 实际长度={actual_length}, response长度={response_len}")
                    # 打印最后几个token的values
                    last_values = all_values[i, -response_len:]
                    print(f"           response部分values: mean={last_values.mean().item():.4f}, "
                          f"min={last_values.min().item():.4f}, max={last_values.max().item():.4f}")
            
            for i in range(batch_size):
                response_length = all_response_lengths[i]
                
                # 提取response部分的values
                values = all_values[i:i+1, -response_length:]  # (1, response_length)
                
                # 获取mask并排除EOS token
                attention_mask = inputs['attention_mask'][i:i+1]
                response_mask = attention_mask[:, -response_length:]
                response_ids = inputs['input_ids'][i:i+1, -response_length:]
                
                eos_token_id = tokenizer.eos_token_id
                is_eos = (response_ids == eos_token_id)
                response_mask_no_eos = response_mask & (~is_eos)
                
                # 🔧 关键修复：理解训练时的分数计算流程
                # 1. _forward_micro_batch返回values_output，shape (batch, response_length)
                #    其中只有最后一个有效位置有平均值，其他位置都是0
                # 2. update_critic中：teacher_score = teacher_vpreds.sum(dim=-1)
                #    因为只有一个位置有值，sum就等于那个平均值
                # 3. compute_discriminator_loss中：
                #    teacher_score_raw = torch.sum(teacher_vpreds * teacher_response_mask, dim=-1)
                #    因为只有最后一个位置有值且mask=1，所以还是平均值
                # 
                # 结论：训练时实际使用的是平均值，不是sum！
                values_sum = (values * response_mask_no_eos).sum(dim=-1)  # (1,)
                length = response_mask_no_eos.sum(dim=-1).clamp(min=1)  # (1,)
                score_avg = (values_sum / length).item()  # 平均值
                
                scores.append(score_avg)
                lengths.append(length.item())
            
            return scores, lengths
            
    except Exception as e:
        print(f"❌ Batch Critic评分失败: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(prompts), [0] * len(prompts)


def test_with_training_batch_structure(
    critic_model,
    tokenizer,
    student_config: dict,
    data_path: str, 
    num_samples: int = 100,
    n_resp_per_prompt: int = 4,
    batch_size: int = 32
):
    """
    使用与训练完全相同的batch结构进行测试
    
    训练时的batch结构：
    - 每个prompt生成n_resp_per_prompt个student responses
    - 一个batch包含batch_size个prompts
    - teachers和students在同一个batch中进行推理
    
    Args:
        critic_model: Critic模型
        tokenizer: Tokenizer
        student_config: Student模型配置
        data_path: 数据集路径
        num_samples: 测试样本数量
        n_resp_per_prompt: 每个prompt生成的student responses数量
        batch_size: 每个batch包含的prompts数量
    """
    # 读取数据
    df = pd.read_parquet(data_path)
    
    # 采样
    if num_samples:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    print(f"📊 测试配置:")
    print(f"   样本数: {len(df)}")
    print(f"   每个prompt的student数: {n_resp_per_prompt}")
    print(f"   Batch大小: {batch_size} prompts")
    print(f"   每个batch总样本数: {batch_size * (1 + n_resp_per_prompt)}")
    print()
    
    # 准备所有数据
    all_prompts = []
    all_teacher_responses = []
    
    for idx, row in df.iterrows():
        try:
            content = row['content']
            if isinstance(content, (list, tuple)) and len(content) > 0:
                prompt = content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
            else:
                prompt = str(content)
            
            teacher_response = row.get('teacher_response', '')
            
            if not prompt or not teacher_response:
                continue
            
            all_prompts.append(prompt)
            all_teacher_responses.append(teacher_response)
        except Exception as e:
            print(f"❌ 解析样本失败: {e}")
            continue
    
    print(f"📝 准备完成，共 {len(all_prompts)} 个有效样本")
    print(f"🔄 开始生成Student responses...")
    
    # 生成所有Student responses
    all_student_responses = []
    for prompt in tqdm(all_prompts, desc="生成responses"):
        student_responses_for_prompt = []
        for _ in range(n_resp_per_prompt):
            response = call_student_api(prompt, student_config)
            if response:
                student_responses_for_prompt.append(response)
            time.sleep(0.1)
        all_student_responses.append(student_responses_for_prompt)
    
    print(f"✅ Student responses生成完成")
    print(f"🔄 开始Batch推理（使用训练时的batch结构）...")
    
    # 使用训练时的batch结构进行推理
    results = []
    total_correct = 0
    total_comparisons = 0
    
    all_teacher_scores = []
    all_student_scores = []
    
    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Batch推理"):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_teacher_responses = all_teacher_responses[batch_start:batch_end]
        batch_student_responses = all_student_responses[batch_start:batch_end]
        
        # 🔧 关键修复：训练时teacher和student是分别forward的，不是混在一起！
        # 所以测试时也应该分别forward
        
        # 1. Forward所有teachers
        teacher_scores, teacher_lengths = get_critic_scores_batch(
            critic_model, tokenizer, batch_prompts, batch_teacher_responses
        )
        
        # 2. Forward所有students（每个prompt有n_resp_per_prompt个students）
        all_student_scores_for_batch = []
        for i, (prompt, student_resps) in enumerate(zip(batch_prompts, batch_student_responses)):
            # 为每个prompt的所有student responses做batch推理
            if len(student_resps) > 0:
                student_scores_for_prompt, _ = get_critic_scores_batch(
                    critic_model, tokenizer, 
                    [prompt] * len(student_resps),  # 重复prompt
                    student_resps
                )
                all_student_scores_for_batch.append(student_scores_for_prompt)
            else:
                all_student_scores_for_batch.append([])
        
        # 3. 计算统计信息
        for i in range(len(batch_prompts)):
            teacher_score = teacher_scores[i]
            student_scores = all_student_scores_for_batch[i]
            
            # 统计
            all_teacher_scores.append(teacher_score)
            all_student_scores.extend(student_scores)
            
            # 计算准确率
            correct = sum(1 for s in student_scores if s <= teacher_score)
            total_correct += correct
            total_comparisons += len(student_scores)
            
            results.append({
                'teacher_score': teacher_score,
                'student_scores': student_scores,
                'correct': correct,
                'total': len(student_scores),
            })
    
    # 打印统计
    print()
    print("=" * 100)
    print("📈 测试统计信息（使用训练时的batch结构）")
    print("=" * 100)
    
    print(f"\n总样本数: {len(results)}")
    print(f"总Student响应数: {total_comparisons}")
    print(f"总体准确率: {total_correct}/{total_comparisons} ({total_correct/total_comparisons*100:.2f}%)")
    print()
    
    print("Teacher模型统计:")
    teacher_mean = sum(all_teacher_scores) / len(all_teacher_scores)
    teacher_std = pd.Series(all_teacher_scores).std()
    print(f"  平均分数: {teacher_mean:.4f}")
    print(f"  分数标准差: {teacher_std:.4f}")
    print(f"  分数范围: [{min(all_teacher_scores):.4f}, {max(all_teacher_scores):.4f}]")
    print()
    
    print("Student模型统计:")
    student_mean = sum(all_student_scores) / len(all_student_scores)
    student_std = pd.Series(all_student_scores).std()
    print(f"  平均分数: {student_mean:.4f}")
    print(f"  分数标准差: {student_std:.4f}")
    print(f"  分数范围: [{min(all_student_scores):.4f}, {max(all_student_scores):.4f}]")
    print()
    
    print("分数差异:")
    score_diff = teacher_mean - student_mean
    print(f"  Teacher - Student: {score_diff:.4f}")
    print()
    
    print("✅ 使用train模式 + 训练时的batch结构")
    print("=" * 100)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="测试Critic模型（使用训练时的batch结构）")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_resp_per_prompt", type=int, default=4,
                       help="每个prompt生成的student responses数量（与训练配置一致）")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="每个batch包含的prompts数量（减小以节省显存）")
    parser.add_argument("--critic_path", type=str, default=None,
                       help="Critic模型路径（可选）")
    parser.add_argument("--use_multi_gpu", action="store_true", default=True,
                       help="使用多GPU（默认开启）")
    parser.add_argument("--gpu_ids", type=str, default="4,5",
                       help="指定使用的GPU IDs，用逗号分隔，例如：'4,5' 或 '0,1,2,3'")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="最大序列长度")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.critic_path:
        CRITIC_CONFIG["model_path"] = args.critic_path
    CRITIC_CONFIG["use_multi_gpu"] = args.use_multi_gpu
    
    print("=" * 100)
    print("📋 测试配置信息（完全模拟训练时的batch结构）")
    print("=" * 100)
    print(f"数据集路径: {args.data_path}")
    print(f"测试样本数: {args.num_samples}")
    print(f"每个prompt的Student样本数: {args.n_resp_per_prompt}")
    print(f"Batch大小: {args.batch_size} prompts")
    print(f"Critic模型: {CRITIC_CONFIG['model_path']}")
    print(f"Student模型: {STUDENT_CONFIG['url']}")
    print(f"多GPU模式: {'开启' if args.use_multi_gpu else '关闭'}")
    if args.use_multi_gpu:
        print(f"指定GPU: {args.gpu_ids} (已通过CUDA_VISIBLE_DEVICES设置)")
    print(f"最大序列长度: {args.max_length}")
    print()
    print("关键改进:")
    print("  1. ✅ 使用train模式（而非eval模式）")
    print("  2. ✅ Teachers和Students在同一个batch中推理")
    print("  3. ✅ 模拟训练时的n_resp_per_prompt=4配置")
    if args.use_multi_gpu:
        print(f"  4. ⚡ 使用多GPU加速（GPU: {args.gpu_ids}）")
    print("=" * 100)
    print()
    
    # 加载模型
    critic_model, tokenizer = load_critic_model(CRITIC_CONFIG)
    
    # 运行测试
    test_with_training_batch_structure(
        critic_model=critic_model,
        tokenizer=tokenizer,
        student_config=STUDENT_CONFIG,
        data_path=args.data_path,
        num_samples=args.num_samples,
        n_resp_per_prompt=args.n_resp_per_prompt,
        batch_size=args.batch_size
    )
    
    print("✅ 测试完成！")


if __name__ == "__main__":
    main()
