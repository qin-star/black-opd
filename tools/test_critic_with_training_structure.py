"""
测试Critic模型 - 完全模拟训练时的batch结构

关键改进：
1. 使用train模式（而非eval模式）
2. 使用与训练完全相同的batch结构（teachers和students在同一个batch中）
3. 模拟训练时的n_resp_per_prompt=4配置
"""

import sys
sys.path.append('/home/jovyan/JQ/gad_gspo_B300/tools')

from test_critic_simplified import CriticTester, API_CONFIGS, TEST_CONFIG
import pandas as pd
import torch
from tqdm import tqdm
import time


class CriticTesterWithTrainingStructure(CriticTester):
    """使用训练时batch结构的Critic测试器"""
    
    def load_critic_model(self, config):
        """加载Critic模型（使用train模式）"""
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = config.get("device", "cuda:0")
        model_path = config["model_path"]
        
        print(f"🔄 加载tokenizer...")
        self.critic_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"🔄 加载模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        
        # 🔧 关键：使用train模式（与训练时一致）
        self.critic_model.train()
        print(f"⚠️  使用train模式（与训练时一致）")
    
    def test_with_training_batch_structure(
        self, 
        data_path: str, 
        num_samples: int = 100,
        n_resp_per_prompt: int = 4,  # 与训练配置一致
        batch_size: int = 32  # 每个batch包含多少个prompts
    ):
        """
        使用与训练完全相同的batch结构进行测试
        
        训练时的batch结构：
        - 每个prompt生成n_resp_per_prompt个student responses
        - 一个batch包含batch_size个prompts
        - teachers和students在同一个batch中进行推理
        
        Args:
            data_path: 数据集路径
            num_samples: 测试样本数量
            n_resp_per_prompt: 每个prompt生成的student responses数量（与训练一致）
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
                response = self.call_student_api(prompt, debug=False)
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
            
            # 🔧 关键：构建训练时的batch结构
            # 格式：[teacher_1, ..., teacher_N, student_1a, student_1b, ..., student_Na, ...]
            
            mixed_prompts = []
            mixed_responses = []
            teacher_indices = []
            student_indices_map = {}
            
            current_idx = 0
            
            # 先添加所有teachers
            for i, (prompt, teacher_resp) in enumerate(zip(batch_prompts, batch_teacher_responses)):
                mixed_prompts.append(prompt)
                mixed_responses.append(teacher_resp)
                teacher_indices.append(current_idx)
                current_idx += 1
            
            # 再添加所有students
            for i, (prompt, student_resps) in enumerate(zip(batch_prompts, batch_student_responses)):
                student_start_idx = current_idx
                for student_resp in student_resps:
                    mixed_prompts.append(prompt)
                    mixed_responses.append(student_resp)
                    current_idx += 1
                student_indices_map[i] = list(range(student_start_idx, current_idx))
            
            # Batch推理（所有teachers和students在同一个batch中）
            all_scores, all_lengths = self.get_critic_scores_batch(
                mixed_prompts, mixed_responses
            )
            
            # 分离teacher和student分数
            for i in range(len(batch_prompts)):
                teacher_score = all_scores[teacher_indices[i]]
                student_score_indices = student_indices_map[i]
                student_scores = [all_scores[idx] for idx in student_score_indices]
                
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
    import argparse
    
    parser = argparse.ArgumentParser(description="测试Critic模型（使用训练时的batch结构）")
    parser.add_argument("--data_path", type=str, 
                       default="/home/jovyan/JQ/gad_gspo_B300/data/trainning_dataset/subject_1-29/merged/merge-1-29.parquet")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_resp_per_prompt", type=int, default=4,
                       help="每个prompt生成的student responses数量（与训练配置一致）")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="每个batch包含的prompts数量")
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("📋 测试配置信息（完全模拟训练时的batch结构）")
    print("=" * 100)
    print(f"数据集路径: {args.data_path}")
    print(f"测试样本数: {args.num_samples}")
    print(f"每个prompt的Student样本数: {args.n_resp_per_prompt}")
    print(f"Batch大小: {args.batch_size} prompts")
    print(f"Critic模型: {API_CONFIGS['critic_model']['model_path']}")
    print(f"Student模型: {API_CONFIGS['student_model']['url']}")
    print()
    print("关键改进:")
    print("  1. ✅ 使用train模式（而非eval模式）")
    print("  2. ✅ Teachers和Students在同一个batch中推理")
    print("  3. ✅ 模拟训练时的n_resp_per_prompt=4配置")
    print("=" * 100)
    print()
    
    tester = CriticTesterWithTrainingStructure(
        critic_config=API_CONFIGS["critic_model"],
        student_config=API_CONFIGS["student_model"],
        test_config=TEST_CONFIG
    )
    
    tester.test_with_training_batch_structure(
        data_path=args.data_path,
        num_samples=args.num_samples,
        n_resp_per_prompt=args.n_resp_per_prompt,
        batch_size=args.batch_size
    )
    
    print("✅ 测试完成！")


if __name__ == "__main__":
    main()
