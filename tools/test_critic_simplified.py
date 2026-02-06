"""
测试训练后的Critic和Student模型 - 简化版（与训练代码完全一致）

与训练代码的一致性：
- 使用平均分数（排除EOS token），与训练代码一致
- 不使用混合分数和batch normalization
- 每个prompt生成8个student responses进行评估
"""
import requests
import pandas as pd
import argparse
from typing import Dict, Optional, List
from datetime import datetime
from tqdm import tqdm
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== 配置区域 ====================

# 测试参数配置
TEST_CONFIG = {
    "data_path": "/home/jovyan/JQ/gad_gspo_B300/data/trainning_dataset/subject_1-29/merged/merge-1-29.parquet",
    "num_samples": 100,
    "random_sample": True,
    "random_seed": 42,
    "output_dir": "/home/jovyan/JQ/gad_gspo_B300/outputs",
    "output_filename": None,
    
    # 多样本生成配置
    "num_student_samples": 8,  # 每个prompt生成8个student response
    "student_temperature": 0.6,  # 提高temperature增加多样性
}

# 模型配置
API_CONFIGS = {
    "critic_model": {
        "name": "critic-model",
        "type": "local",
        "model_path": "/home/jovyan/JQ/gad_gspo_B300/models/opd-v9-1-29-fsdp2/global_step_500/critic_merged",
        "device": "cuda:4",
        "force_trl": True,
    },
    "student_model": {
        "name": "student-model",
        "type": "api",
        "url": "http://10.72.1.39:8008/v1/chat/completions",
        "api_key": "sk-xxxx",
        "model_name": "opd-v9-500",
        "temperature": 0.6,
        "repetition_penalty": 1.2
    }
}
# ================================================



class CriticTester:
    """简化版Critic测试器 - 与训练代码完全一致"""
    
    def __init__(self, critic_config: Dict, student_config: Dict, test_config: Dict):
        self.critic_config = critic_config
        self.student_config = student_config
        self.test_config = test_config
        
        # 加载Critic模型
        self.critic_model = None
        self.critic_tokenizer = None
        if critic_config.get("type") == "local":
            print(f"🔄 加载本地Critic模型: {critic_config['model_path']}")
            self.load_critic_model(critic_config)
            print(f"✅ Critic模型加载完成")
    
    def load_critic_model(self, config: Dict):
        """加载Critic模型"""
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoModelForCausalLM
        
        device = config.get("device", "cuda:0")
        model_path = config["model_path"]
        use_train_mode = config.get("use_train_mode", False)
        
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
        
        # 根据配置选择train或eval模式
        if use_train_mode:
            self.critic_model.train()
            print(f"⚠️  使用train模式（与训练时一致）")
        else:
            self.critic_model.eval()
            print(f"✅ 使用eval模式")
    
    def call_student_api(self, prompt: str, debug: bool = False) -> str:
        """调用Student API生成response"""
        try:
            config = self.student_config
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": config["model_name"],
                "messages": messages,
                "max_tokens": 512,
                "temperature": config.get("temperature", 0.8),
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
            
            # 只移除首尾空白，保持原始输出
            cleaned_content = raw_content.strip()
            
            # 调试模式：显示原始输出
            if debug and raw_content != cleaned_content:
                print(f"  ⚠️  检测到首尾空白字符:")
                print(f"    原始: {repr(raw_content)}")
                print(f"    清理后: {repr(cleaned_content)}")
            
            return cleaned_content
        except Exception as e:
            print(f"❌ Student API调用失败: {e}")
            return ""
    
    def get_critic_score(self, prompt: str, response: str) -> tuple:
        """
        获取Critic分数（单样本，与训练代码完全一致）
        
        Returns:
            (score_avg, length): 平均分数和response长度（排除EOS）
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            input_text = self.critic_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            inputs = self.critic_tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=2048
            )
            
            device = next(self.critic_model.pretrained_model.parameters()).device
            inputs = inputs.to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.critic_model(**inputs, use_cache=False)
                
                # 获取response长度
                response_tokens = self.critic_tokenizer(
                    response, add_special_tokens=False, return_tensors="pt"
                )
                response_length = response_tokens['input_ids'].size(1)
                
                # 提取response部分的values
                values = outputs[2][:, -response_length:]
                if values.dim() == 3:
                    values = values.squeeze(-1)
                
                # 获取mask并排除EOS token
                attention_mask = inputs['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                response_ids = inputs['input_ids'][:, -response_length:]
                
                eos_token_id = self.critic_tokenizer.eos_token_id
                is_eos = (response_ids == eos_token_id)
                response_mask_no_eos = response_mask & (~is_eos)
                
                # 计算平均分数（与训练代码一致）
                values_sum = torch.sum(values * response_mask_no_eos, dim=-1)
                length = response_mask_no_eos.sum(dim=-1)
                score_avg = (values_sum / length.clamp(min=1)).item()
                length = length.item()
                
                return score_avg, length
        except Exception as e:
            print(f"❌ Critic评分失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0
    
    def get_critic_scores_batch(self, prompts: list, responses: list) -> tuple:
        """
        获取Critic分数（批量推理，与训练代码一致）
        
        Args:
            prompts: prompt列表
            responses: response列表
        
        Returns:
            (scores, lengths): 分数列表和长度列表
        """
        try:
            device = next(self.critic_model.pretrained_model.parameters()).device
            batch_size = len(prompts)
            
            # 准备batch数据
            all_input_texts = []
            all_response_lengths = []
            
            for prompt, response in zip(prompts, responses):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                
                input_text = self.critic_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                all_input_texts.append(input_text)
                
                # 计算response长度
                response_tokens = self.critic_tokenizer(
                    response, add_special_tokens=False, return_tensors="pt"
                )
                all_response_lengths.append(response_tokens['input_ids'].size(1))
            
            # Batch tokenization（使用padding）
            inputs = self.critic_tokenizer(
                all_input_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True  # 关键：添加padding
            )
            inputs = inputs.to(device)
            
            # Batch前向传播
            with torch.no_grad():
                outputs = self.critic_model(**inputs, use_cache=False)
                
                # 提取每个样本的values
                scores = []
                lengths = []
                
                for i in range(batch_size):
                    response_length = all_response_lengths[i]
                    
                    # 提取response部分的values
                    values = outputs[2][i:i+1, -response_length:]
                    if values.dim() == 3:
                        values = values.squeeze(-1)
                    
                    # 获取mask并排除EOS token
                    attention_mask = inputs['attention_mask'][i:i+1]
                    response_mask = attention_mask[:, -response_length:]
                    response_ids = inputs['input_ids'][i:i+1, -response_length:]
                    
                    eos_token_id = self.critic_tokenizer.eos_token_id
                    is_eos = (response_ids == eos_token_id)
                    response_mask_no_eos = response_mask & (~is_eos)
                    
                    # 计算平均分数
                    values_sum = torch.sum(values * response_mask_no_eos, dim=-1)
                    length = response_mask_no_eos.sum(dim=-1)
                    score_avg = (values_sum / length.clamp(min=1)).item()
                    
                    scores.append(score_avg)
                    lengths.append(length.item())
                
                return scores, lengths
                
        except Exception as e:
            print(f"❌ Batch Critic评分失败: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(prompts), [0] * len(prompts)

    
    def test_single_prompt(self, prompt: str, teacher_response: str, debug: bool = False) -> Dict:
        """测试单个prompt（生成多个student responses）
        
        注意：正确率判断使用 student_score <= teacher_score（包括相等情况）
        """
        print("📝 Prompt:")
        print(f"  {prompt}")
        print()
        
        num_students = self.test_config.get("num_student_samples", 8)
        
        # 生成多个Student响应
        print(f"🔄 生成 {num_students} 个Student响应...")
        student_responses = []
        for i in range(num_students):
            response = self.call_student_api(prompt, debug=debug)
            if response:
                student_responses.append(response)
                print(f"  ✓ Student #{i+1}: {len(response)} chars")
        print()
        
        if len(student_responses) == 0:
            print("❌ 没有成功生成student responses")
            return None
        
        # 获取Teacher分数
        teacher_score, teacher_length = self.get_critic_score(prompt, teacher_response)
        
        # 获取所有Student分数
        student_scores = []
        student_lengths = []
        for response in student_responses:
            score, length = self.get_critic_score(prompt, response)
            student_scores.append(score)
            student_lengths.append(length)
        
        # 打印结果
        print("👨‍🏫 Teacher Response:")
        print("-" * 100)
        print(f"  分数(avg): {teacher_score:7.4f}")
        print(f"  Length: {int(teacher_length):3d}")
        print(f"  Text: {teacher_response}")
        if debug:
            print(f"  Repr: {repr(teacher_response)}")
        print()
        
        print(f"🎓 Student Responses ({len(student_responses)} 个):")
        print("-" * 100)
        for i, (response, score, length) in enumerate(zip(student_responses, student_scores, student_lengths)):
            diff = score - teacher_score
            # 修改判断逻辑：student_score <= teacher_score 为正确（包括相等情况）
            is_correct = score <= teacher_score
            status = '✅正确' if is_correct else '❌错误'
            
            # 特殊标记相等情况
            if abs(diff) < 1e-6:  # 浮点数相等判断
                status += ' (相等)'
            
            print(f"\n  Student #{i+1}:")
            print(f"    分数(avg): {score:7.4f}")
            print(f"    Length: {int(length):3d}")
            print(f"    与Teacher分差: {diff:7.4f} ({status})")
            print(f"    Text: {response}")
            
            # 调试模式：显示详细的字符差异
            if debug and response != teacher_response:
                print(f"    Repr: {repr(response)}")
                # 检查是否只是空白字符差异
                if response.replace(' ', '').replace('\n', '').replace('\t', '') == \
                   teacher_response.replace(' ', '').replace('\n', '').replace('\t', ''):
                    print(f"    ⚠️  与Teacher仅空白字符不同")
        
        # 统计（修改判断逻辑：包括相等情况）
        correct_count = sum(1 for score in student_scores if score <= teacher_score)
        accuracy = correct_count / len(student_scores)
        
        print()
        print("📊 统计信息:")
        print(f"  Teacher分数: {teacher_score:7.4f}")
        print(f"  Student平均分数: {sum(student_scores)/len(student_scores):7.4f}")
        print(f"  准确率: {correct_count}/{len(student_scores)} ({accuracy*100:.1f}%)")
        print(f"  注：正确率判断使用 student_score <= teacher_score（包括相等情况）")
        print()
        print("=" * 100)
        print()
        
        return {
            'prompt': prompt,
            'teacher_response': teacher_response,
            'teacher_score': teacher_score,
            'teacher_length': teacher_length,
            'student_responses': student_responses,
            'student_scores': student_scores,
            'student_lengths': student_lengths,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'num_students': len(student_responses),
        }
    
    def test_dataset(self, data_path: str, num_samples: Optional[int] = None,
                    output_path: Optional[str] = None, use_batch_inference: bool = True,
                    batch_size: int = 32) -> pd.DataFrame:
        """测试数据集
        
        Args:
            data_path: 数据集路径
            num_samples: 测试样本数量
            output_path: 输出路径
            use_batch_inference: 是否使用batch推理（推荐True，与训练一致）
            batch_size: batch推理的批次大小
        """
        # 读取数据
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("仅支持.parquet格式")
        
        # 采样
        if num_samples:
            if self.test_config.get("random_sample"):
                df = df.sample(n=min(num_samples, len(df)), 
                             random_state=self.test_config.get("random_seed", 42))
            else:
                df = df.head(num_samples)
        
        print(f"📊 测试 {len(df)} 个样本...")
        print(f"   推理模式: {'Batch推理' if use_batch_inference else '单样本推理'}")
        if use_batch_inference:
            print(f"   Batch大小: {batch_size}")
        print()
        
        if use_batch_inference:
            results = self._test_dataset_batch(df, batch_size)
        else:
            results = self._test_dataset_single(df)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 打印统计
        self.print_statistics(results_df, use_batch_inference)
        
        # 保存
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "batch" if use_batch_inference else "single"
            output_path = f"{self.test_config['output_dir']}/test_results_{mode}_{timestamp}.xlsx"
        
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_excel(output_path, index=False)
        print(f"✅ 结果已保存到: {output_path}")
        
        return results_df
    
    def _test_dataset_single(self, df: pd.DataFrame) -> list:
        """单样本推理模式"""
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="测试进度（单样本）"):
            try:
                # 解析数据
                content = row['content']
                if isinstance(content, (list, tuple)) and len(content) > 0:
                    prompt = content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
                else:
                    prompt = str(content)
                
                teacher_response = row.get('teacher_response', '')
                
                if not prompt or not teacher_response:
                    continue
                
                result = self.test_single_prompt(prompt, teacher_response, 
                                                debug=self.test_config.get('debug', False))
                if result:
                    result['sample_id'] = idx
                    results.append(result)
                
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ 测试样本 {idx} 失败: {e}")
                continue
        
        return results
    
    def _test_dataset_batch(self, df: pd.DataFrame, batch_size: int = 32) -> list:
        """Batch推理模式（与训练一致）"""
        results = []
        
        # 准备所有数据
        all_prompts = []
        all_teacher_responses = []
        all_sample_ids = []
        
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
                all_sample_ids.append(idx)
            except Exception as e:
                print(f"❌ 解析样本 {idx} 失败: {e}")
                continue
        
        print(f"📝 准备完成，共 {len(all_prompts)} 个有效样本")
        print(f"🔄 开始生成Student responses...")
        
        # 生成所有Student responses
        num_students = self.test_config.get("num_student_samples", 8)
        all_student_responses = []
        
        for i, prompt in enumerate(tqdm(all_prompts, desc="生成responses")):
            student_responses_for_prompt = []
            for _ in range(num_students):
                response = self.call_student_api(prompt, debug=False)
                if response:
                    student_responses_for_prompt.append(response)
                time.sleep(0.1)
            all_student_responses.append(student_responses_for_prompt)
        
        print(f"✅ Student responses生成完成")
        print(f"🔄 开始Batch推理...")
        
        # Batch推理
        total_samples = len(all_prompts)
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Batch推理"):
            batch_end = min(batch_start + batch_size, total_samples)
            
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_teacher_responses = all_teacher_responses[batch_start:batch_end]
            batch_student_responses = all_student_responses[batch_start:batch_end]
            batch_sample_ids = all_sample_ids[batch_start:batch_end]
            
            # 获取Teacher分数（batch）
            teacher_scores, teacher_lengths = self.get_critic_scores_batch(
                batch_prompts, batch_teacher_responses
            )
            
            # 对每个prompt的多个student responses进行评分
            for i in range(len(batch_prompts)):
                prompt = batch_prompts[i]
                teacher_response = batch_teacher_responses[i]
                teacher_score = teacher_scores[i]
                teacher_length = teacher_lengths[i]
                student_responses = batch_student_responses[i]
                
                if len(student_responses) == 0:
                    continue
                
                # 获取Student分数（batch）
                student_scores, student_lengths = self.get_critic_scores_batch(
                    [prompt] * len(student_responses),
                    student_responses
                )
                
                # 计算准确率
                correct_count = sum(1 for score in student_scores if score <= teacher_score)
                accuracy = correct_count / len(student_scores)
                
                # 保存结果
                results.append({
                    'sample_id': batch_sample_ids[i],
                    'prompt': prompt,
                    'teacher_response': teacher_response,
                    'teacher_score': teacher_score,
                    'teacher_length': teacher_length,
                    'student_responses': student_responses,
                    'student_scores': student_scores,
                    'student_lengths': student_lengths,
                    'correct_count': correct_count,
                    'accuracy': accuracy,
                    'num_students': len(student_responses),
                })
        
        return results
    
    def print_statistics(self, results_df: pd.DataFrame, use_batch_inference: bool = False):
        """打印统计信息"""
        print()
        print("=" * 100)
        print(f"📈 测试统计信息（{'Batch推理' if use_batch_inference else '单样本推理'} - 与训练代码一致）")
        print("=" * 100)
        
        total = len(results_df)
        if total == 0:
            print("⚠️  没有成功测试的样本")
            return
        
        total_students = results_df['num_students'].sum()
        total_correct = results_df['correct_count'].sum()
        avg_accuracy = results_df['accuracy'].mean() * 100
        overall_accuracy = total_correct / total_students * 100
        
        print(f"总样本数: {total}")
        print(f"总Student响应数: {total_students}")
        print(f"平均准确率: {avg_accuracy:.2f}%")
        print(f"总体准确率: {total_correct}/{total_students} ({overall_accuracy:.2f}%)")
        print(f"注：正确率判断使用 student_score <= teacher_score（包括相等情况）")
        print()
        
        print("Teacher模型统计:")
        print(f"  平均分数: {results_df['teacher_score'].mean():.4f}")
        print(f"  分数标准差: {results_df['teacher_score'].std():.4f}")
        print(f"  分数范围: [{results_df['teacher_score'].min():.4f}, {results_df['teacher_score'].max():.4f}]")
        print(f"  平均长度: {results_df['teacher_length'].mean():.2f}")
        print()
        
        # Student统计（需要展开列表）
        all_student_scores = []
        all_student_lengths = []
        for _, row in results_df.iterrows():
            all_student_scores.extend(row['student_scores'])
            all_student_lengths.extend(row['student_lengths'])
        
        print("Student模型统计:")
        print(f"  平均分数: {sum(all_student_scores)/len(all_student_scores):.4f}")
        print(f"  分数标准差: {pd.Series(all_student_scores).std():.4f}")
        print(f"  分数范围: [{min(all_student_scores):.4f}, {max(all_student_scores):.4f}]")
        print(f"  平均长度: {sum(all_student_lengths)/len(all_student_lengths):.2f}")
        print()
        
        # 分数差异统计
        score_diff = results_df['teacher_score'].mean() - sum(all_student_scores)/len(all_student_scores)
        print("分数差异:")
        print(f"  Teacher - Student: {score_diff:.4f}")
        
        if use_batch_inference:
            print()
            print("✅ 使用Batch推理，与训练时的推理方式一致")
        else:
            print()
            print("⚠️  使用单样本推理，与训练时的推理方式不同")
            print("   建议使用 --use_batch_inference 参数进行测试")
        
        print()
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="测试Critic模型（简化版）")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="启用调试模式，显示详细的字符差异")
    parser.add_argument("--use_batch_inference", action="store_true", default=True,
                       help="使用batch推理（推荐，与训练一致）")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch推理的批次大小")
    parser.add_argument("--no_batch", action="store_true",
                       help="禁用batch推理，使用单样本推理")
    
    args = parser.parse_args()
    
    data_path = args.data_path or TEST_CONFIG["data_path"]
    num_samples = args.num_samples if args.num_samples is not None else TEST_CONFIG["num_samples"]
    output_path = args.output_path
    debug = args.debug
    use_batch_inference = not args.no_batch  # 默认使用batch推理
    batch_size = args.batch_size
    
    print("=" * 100)
    print("📋 测试配置信息（简化版 - 与训练代码一致）")
    print("=" * 100)
    print(f"数据集路径: {data_path}")
    print(f"测试样本数: {num_samples if num_samples else '全部'}")
    print(f"每个prompt的Student样本数: {TEST_CONFIG['num_student_samples']}")
    print(f"Critic模型: {API_CONFIGS['critic_model']['model_path']}")
    print(f"Student模型: {API_CONFIGS['student_model']['url']}")
    print(f"调试模式: {'开启' if debug else '关闭'}")
    print(f"推理模式: {'Batch推理' if use_batch_inference else '单样本推理'}")
    if use_batch_inference:
        print(f"Batch大小: {batch_size}")
    print()
    print("分数计算方式: 平均分数（排除EOS token），与训练代码完全一致")
    if use_batch_inference:
        print("✅ 使用Batch推理，与训练时的推理方式一致（推荐）")
    else:
        print("⚠️  使用单样本推理，与训练时的推理方式不同")
    print("=" * 100)
    print()
    
    tester = CriticTester(
        critic_config=API_CONFIGS["critic_model"],
        student_config=API_CONFIGS["student_model"],
        test_config=TEST_CONFIG
    )
    
    # 传递debug参数
    TEST_CONFIG['debug'] = debug
    
    tester.test_dataset(
        data_path, 
        num_samples, 
        output_path,
        use_batch_inference=use_batch_inference,
        batch_size=batch_size
    )
    print("✅ 测试完成！")


if __name__ == "__main__":
    main()
