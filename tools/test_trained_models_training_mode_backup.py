"""
测试训练后的Critic和Student模型 - 训练模式
完全复现训练时的分数计算方式（包括混合归一化、batch normalization、temperature缩放）

与 test_trained_models.py 的区别：
- test_trained_models.py: 使用原始的value head输出（模型真实能力）
- 本脚本: 使用训练时的归一化和缩放（复现训练日志中的分数）
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
# 在这里修改你的模型配置和测试参数

# 测试参数配置
TEST_CONFIG = {
    "data_path": "/home/jovyan/JQ/gad_gspo_B300/data/trainning_dataset/subject_1-29/merged/merge-1-29.parquet",
    "num_samples": 10,  # 测试样本数量（None表示全部）
    "random_sample": True,  # 是否随机抽样
    "random_seed": 42,  # 随机种子
    "output_dir": "/home/jovyan/JQ/gad_gspo_B300/outputs",
    "output_filename": None,  # None表示自动生成
    
    # 多样本生成配置
    "num_student_samples": 8,  # 每个prompt生成多少个student response
    "student_temperature": 0.8,  # Student生成的temperature（增加多样性）
    
    # 训练模式配置（与简化后的训练代码保持一致）
    # 注意：训练代码已简化，直接使用原始分数（sum），不再使用混合分数和batch norm
    "use_raw_score_only": True,  # ✅ 只使用原始分数（sum），与训练代码一致
    "temperature": 5.0,  # Temperature缩放系数（仅用于loss计算，不影响分数本身）
}

# 模型配置
API_CONFIGS = {
    "critic_model": {
        "name": "critic-model",
        "type": "local",
        "model_path": "/home/jovyan/JQ/gad_gspo_B300/models/opd-v9-1-29-fsdp2/global_step_500/critic_merged",
        "device": "cuda:4",
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "force_trl": True,
    },
    "student_model": {
        "name": "student-model",
        "type": "api",
        "url": "http://10.72.1.39:8008/v1/chat/completions",
        "api_key": "sk-xxxx",
        "model_name": "opd-v9-500",
        "temperature": 0.8,  # 提高temperature以增加多样性
        "repetition_penalty": 1.2
    }
}
# ================================================


class ModelTesterTrainingMode:
    """
    训练模式的模型测试器
    完全复现训练时的分数计算逻辑
    """
    def __init__(self, critic_config: Dict, student_config: Dict, training_config: Dict):
        self.critic_config = critic_config
        self.student_config = student_config
        self.training_config = training_config
        
        # 加载本地Critic模型
        self.critic_model = None
        self.critic_tokenizer = None
        if critic_config.get("type") == "local":
            print(f"🔄 加载本地Critic模型: {critic_config['model_path']}")
            self.load_local_critic(critic_config)
            print(f"✅ Critic模型加载完成")
    
    def load_local_critic(self, config: Dict):
        """加载本地Critic模型"""
        device = config.get("device", "cuda:0")
        model_path = config["model_path"]
        force_trl = config.get("force_trl", False)
        
        print(f"🔄 加载tokenizer...")
        self.critic_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer加载完成")
        
        if force_trl:
            print(f"⚡ 配置强制使用trl，跳过TokenClassification尝试")
            self._load_with_trl(model_path, device)
            return
        
        from transformers import AutoModelForTokenClassification, AutoConfig
        
        print(f"🔍 检查模型配置...")
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  架构: {model_config.architectures}")
        
        is_token_classification = any('TokenClassification' in arch for arch in model_config.architectures)
        
        if is_token_classification:
            print(f"🔄 使用 AutoModelForTokenClassification 加载...")
            try:
                self.critic_model = AutoModelForTokenClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map=device
                )
                print(f"✅ 成功使用 AutoModelForTokenClassification 加载")
                self.critic_model.eval()
                return
            except Exception as e:
                print(f"❌ AutoModelForTokenClassification 加载失败: {e}")
                raise
        
        print(f"🔄 模型不是TokenClassification类型，使用trl加载...")
        self._load_with_trl(model_path, device)
    
    def _load_with_trl(self, model_path: str, device: str):
        """使用trl加载模型"""
        try:
            from trl import AutoModelForCausalLMWithValueHead
            from transformers import AutoModelForCausalLM
            
            print(f"  加载基础CausalLM模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device
            )
            
            print(f"  包装为ValueHead模型...")
            self.critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
            print(f"✅ 成功使用 trl.AutoModelForCausalLMWithValueHead 加载")
            self.critic_model.eval()
        except Exception as e:
            print(f"❌ trl加载失败: {e}")
            raise RuntimeError(f"无法加载Critic模型: {e}")

    
    def call_generation_model(self, config: Dict, prompt: str, max_tokens: int = 512) -> Dict:
        """调用生成模型API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": config["model_name"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": config.get("temperature", 0.7),
                "top_p": 0.9,
                "repetition_penalty": config.get("repetition_penalty", 1.0),
                "n": 1
            }
            
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                config["url"],
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            tokens = usage.get('completion_tokens', len(generated_text.split()))
            
            return {
                'text': generated_text.strip(),
                'length': tokens
            }
        except Exception as e:
            print(f"❌ 生成模型调用失败 [{config['name']}]: {e}")
            return {'text': '', 'length': 0}
    
    def get_raw_values(self, prompt: str, response: str) -> tuple:
        """
        获取原始的value head输出（未经归一化和缩放）
        
        Returns:
            values: 原始values tensor, shape (1, response_length)
            response_mask: response mask tensor, shape (1, response_length)
            response_length: response长度
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            if hasattr(self.critic_tokenizer, 'apply_chat_template'):
                input_text = self.critic_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                input_text = f"User: {prompt}\nAssistant: {response}"
            
            inputs = self.critic_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # 获取设备
            if hasattr(self.critic_model, 'device'):
                device = self.critic_model.device
            elif hasattr(self.critic_model, 'pretrained_model'):
                device = next(self.critic_model.pretrained_model.parameters()).device
            else:
                device = next(self.critic_model.parameters()).device
            
            inputs = inputs.to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # 前向传播
            with torch.no_grad():
                outputs = self.critic_model(**inputs, use_cache=False)
                
                # 获取values
                if hasattr(outputs, 'logits'):
                    # TokenClassification模型
                    response_tokens = self.critic_tokenizer(
                        response,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )
                    response_length = response_tokens['input_ids'].size(1)
                    values = outputs.logits[:, -response_length:]
                    if values.dim() == 3:
                        values = values.squeeze(-1)
                elif hasattr(self.critic_model, 'v_head') or isinstance(outputs, tuple):
                    # trl模型
                    if isinstance(outputs, tuple) and len(outputs) > 2:
                        response_tokens = self.critic_tokenizer(
                            response,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )
                        response_length = response_tokens['input_ids'].size(1)
                        values = outputs[2][:, -response_length:]
                        if values.dim() == 3:
                            values = values.squeeze(-1)
                    else:
                        raise ValueError("模型返回tuple但格式不正确")
                else:
                    raise ValueError(f"无法识别的模型输出格式: {type(outputs)}")
            
            response_mask = attention_mask[:, -response_length:]
            response_ids = input_ids[:, -response_length:]
            
            # 排除EOS token
            eos_token_id = self.critic_tokenizer.eos_token_id
            is_eos = (response_ids == eos_token_id)
            response_mask_no_eos = response_mask & (~is_eos)
            
            return values, response_mask_no_eos, response_length
            
        except Exception as e:
            print(f"❌ 获取原始values失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0

    
    def compute_training_mode_scores_batch(self,
                                           teacher_values: torch.Tensor,
                                           student_values_list: List[torch.Tensor],
                                           teacher_mask: torch.Tensor,
                                           student_masks_list: List[torch.Tensor]) -> Dict:
        """
        使用训练时的方式计算分数（批量版本 - 简化版）
        完全复现简化后的 compute_discriminator_loss 中的分数计算逻辑
        
        训练代码已简化：
        - 直接使用原始分数（sum），不再使用混合分数
        - 不再使用batch normalization
        - 分数差异直接计算，temperature仅用于loss
        
        Args:
            teacher_values: shape (1, response_length)
            student_values_list: list of tensors, each shape (1, response_length)
            teacher_mask: shape (1, response_length)
            student_masks_list: list of tensors, each shape (1, response_length)
        
        Returns:
            包含各种分数的字典
        """
        # ==============================
        # 1. 计算原始分数（sum）- 与训练代码完全一致
        # ==============================
        eps = 1e-8
        
        # Teacher分数（原始sum）
        teacher_score_raw = torch.sum(teacher_values * teacher_mask, dim=-1)
        teacher_mask_sum = teacher_mask.sum(dim=-1).clamp(min=eps)
        
        # 归一化分数（仅用于参考，不用于判断）
        teacher_score_norm = teacher_score_raw / teacher_mask_sum
        
        # Student分数（多个）
        student_scores_raw = []
        student_scores_norm = []
        student_lengths = []
        
        for student_values, student_mask in zip(student_values_list, student_masks_list):
            score_raw = torch.sum(student_values * student_mask, dim=-1)
            mask_sum = student_mask.sum(dim=-1).clamp(min=eps)
            score_norm = score_raw / mask_sum
            
            student_scores_raw.append(score_raw)
            student_scores_norm.append(score_norm)
            student_lengths.append(mask_sum)
        
        # 转换为tensor
        student_scores_raw = torch.stack(student_scores_raw)  # (num_students,)
        student_scores_norm = torch.stack(student_scores_norm)  # (num_students,)
        student_lengths = torch.stack(student_lengths)  # (num_students,)
        
        # ==============================
        # 2. 使用原始分数（与训练代码一致）
        # ==============================
        # 训练代码简化：teacher_score = teacher_score_raw
        teacher_score = teacher_score_raw
        student_scores = student_scores_raw
        
        # ==============================
        # 3. 计算分数差异（直接计算，不使用batch norm）
        # ==============================
        # 训练代码：diff = teacher_score - student_score
        diffs = student_scores - teacher_score  # 注意：这里是student - teacher
        
        # Temperature缩放（仅用于loss计算，不影响判断）
        temperature = self.training_config.get("temperature", 5.0)
        diffs_scaled = diffs / temperature
        
        # ==============================
        # 4. 返回所有分数信息
        # ==============================
        return {
            # Teacher分数
            "teacher_score_raw": teacher_score_raw.item(),
            "teacher_score_norm": teacher_score_norm.item(),
            "teacher_score": teacher_score.item(),  # 最终使用的分数
            "teacher_length": teacher_mask_sum.item(),
            
            # Student分数（列表）
            "student_scores_raw": student_scores_raw.cpu().tolist(),
            "student_scores_norm": student_scores_norm.cpu().tolist(),
            "student_scores": student_scores.cpu().tolist(),  # 最终使用的分数
            "student_lengths": student_lengths.cpu().tolist(),
            
            # 分数差异（列表）- 注意：负值表示student < teacher（正确）
            "score_diffs": diffs.cpu().tolist(),
            "score_diffs_scaled": diffs_scaled.cpu().tolist(),
            
            # 统计信息
            "student_score_mean": student_scores.mean().item(),
            "student_score_norm_mean": student_scores_norm.mean().item(),
            "student_length_mean": student_lengths.mean().item(),
            
            "score_diff_mean": diffs.mean().item(),
            "score_diff_scaled_mean": diffs_scaled.mean().item(),
            
            # 配置信息
            "temperature": temperature,
            "use_raw_score_only": True,
            "num_students": len(student_values_list),
        }

    
    def test_single_sample(self, prompt: str, teacher_response: str) -> Dict:
        """测试单个样本（生成多个student responses）"""
        print("📝 Prompt:")
        print(f"  {prompt}")
        print()
        
        num_students = self.training_config.get("num_student_samples", 8)
        
        # 生成多个Student响应
        print(f"🔄 生成 {num_students} 个Student响应...")
        student_results = []
        for i in range(num_students):
            result = self.call_generation_model(self.student_config, prompt)
            student_results.append(result)
            print(f"  ✓ Student #{i+1}: {len(result['text'])} chars")
        print()
        
        # 获取teacher的原始values
        teacher_values, teacher_mask, teacher_length = self.get_raw_values(prompt, teacher_response)
        
        if teacher_values is None:
            print("❌ 无法获取teacher values，跳过此样本")
            return None
        
        # 获取所有student的原始values
        student_values_list = []
        student_masks_list = []
        
        for i, result in enumerate(student_results):
            values, mask, length = self.get_raw_values(prompt, result['text'])
            if values is None:
                print(f"❌ 无法获取Student #{i+1} values，跳过")
                continue
            student_values_list.append(values)
            student_masks_list.append(mask)
        
        if len(student_values_list) == 0:
            print("❌ 没有有效的student values，跳过此样本")
            return None
        
        # 计算训练模式分数（批量版本）
        scores = self.compute_training_mode_scores_batch(
            teacher_values, student_values_list,
            teacher_mask, student_masks_list
        )
        
        # 打印结果
        print("�‍🏫 Teacher Response:")
        print("-" * 100)
        print(f"  原始分数(sum):     {scores['teacher_score_raw']:7.4f}")
        print(f"  归一化分数(mean):  {scores['teacher_score_norm']:7.4f}")
        print(f"  混合分数(70/30):   {scores['teacher_score_mixed']:7.4f}")
        print(f"  最终分数(+norm):   {scores['teacher_score_final']:7.4f}")
        print(f"  Length: {int(scores['teacher_length']):3d}")
        print(f"  Text (完整): {teacher_response}")
        print()
        
        print(f"🎓 Student Responses ({len(student_results)} 个):")
        print("-" * 100)
        
        # 显示每个student response及其分数
        for i in range(len(student_values_list)):
            print(f"\n  Student #{i+1}:")
            print(f"    原始分数(sum):     {scores['student_scores_raw'][i]:7.4f}")
            print(f"    归一化分数(mean):  {scores['student_scores_norm'][i]:7.4f}")
            print(f"    混合分数(70/30):   {scores['student_scores_mixed'][i]:7.4f}")
            print(f"    最终分数(+norm):   {scores['student_scores_final'][i]:7.4f}")
            print(f"    Length: {int(scores['student_lengths'][i]):3d}")
            print(f"    与Teacher分差(混合): {scores['score_diffs_mixed'][i]:7.4f}")
            print(f"    与Teacher分差(最终): {scores['score_diffs_final'][i]:7.4f}")
            print(f"    Text (完整): {student_results[i]['text']}")
        
        print()
        print("📊 统计信息:")
        print(f"  Student平均分数(混合): {scores['student_score_mixed_mean']:7.4f}")
        print(f"  Student平均分数(最终): {scores['student_score_final_mean']:7.4f}")
        print(f"  平均分差(混合): {scores['score_diff_mixed_mean']:7.4f}")
        print(f"  平均分差(最终): {scores['score_diff_final_mean']:7.4f}")
        print(f"  平均分差(缩放): {scores['score_diff_scaled_mean']:7.4f} (temperature={scores['temperature']})")
        print()
        
        # 判断准确性
        # 使用混合分数：有多少个student的分数低于teacher
        correct_count_mixed = sum(1 for diff in scores['score_diffs_mixed'] if diff < 0)
        correct_count_final = sum(1 for diff in scores['score_diffs_final'] if diff < 0)
        
        print(f"  Teacher > Student 的比例:")
        print(f"    混合分数: {correct_count_mixed}/{len(student_values_list)} ({correct_count_mixed/len(student_values_list)*100:.1f}%)")
        print(f"    最终分数: {correct_count_final}/{len(student_values_list)} ({correct_count_final/len(student_values_list)*100:.1f}%)")
        
        if scores.get('use_batch_norm'):
            print(f"\n  Batch Normalization:")
            print(f"    Batch size: {scores.get('batch_size', 'N/A')}")
            print(f"    Batch mean: {scores.get('batch_mean', 0):.4f}")
            print(f"    Batch std:  {scores.get('batch_std', 0):.4f}")
        
        print()
        print("=" * 100)
        print()
        
        # 保存结果
        result = {
            'prompt': prompt,
            'teacher_response': teacher_response,
            'num_students': len(student_values_list),
            'correct_count_mixed': correct_count_mixed,
            'correct_count_final': correct_count_final,
            'accuracy_mixed': correct_count_mixed / len(student_values_list),
            'accuracy_final': correct_count_final / len(student_values_list),
            **scores
        }
        
        # 添加每个student的response文本
        for i, student_result in enumerate(student_results[:len(student_values_list)]):
            result[f'student_{i+1}_response'] = student_result['text']
        
        return result

    
    def test_dataset(self, data_path: str, num_samples: Optional[int] = None,
                    output_path: Optional[str] = None, random_sample: bool = False,
                    random_seed: int = 42) -> pd.DataFrame:
        """测试数据集"""
        # 读取数据
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的文件格式，仅支持.parquet和.xlsx")
        
        # 限制测试数量
        if num_samples is not None:
            if random_sample:
                df = df.sample(n=min(num_samples, len(df)), random_state=random_seed)
                print(f"📊 随机抽样 {len(df)} 个样本（种子={random_seed}）...")
            else:
                df = df.head(num_samples)
                print(f"📊 取前 {len(df)} 个样本...")
        else:
            print(f"📊 测试全部 {len(df)} 个样本...")
        print()
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="测试进度"):
            try:
                # 解析content字段
                if 'content' in row:
                    content = row['content']
                    if isinstance(content, (list, tuple)) and len(content) > 0:
                        prompt = content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
                    else:
                        prompt = str(content)
                elif 'prompt' in row:
                    prompt = row['prompt']
                else:
                    print(f"⚠️  跳过第 {idx} 行：无法识别的数据格式")
                    continue
                
                teacher_response = row.get('teacher_response', '')
                
                if not prompt or not teacher_response:
                    print(f"⚠️  跳过第 {idx} 行：缺少prompt或teacher_response")
                    continue
                
                result = self.test_single_sample(prompt, teacher_response)
                if result is not None:
                    result['sample_id'] = idx
                    result['data_id'] = row.get('id', idx)
                    results.append(result)
                
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ 测试第 {idx} 个样本时出错: {e}")
                continue
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 打印统计信息
        self.print_statistics(results_df)
        
        # 保存结果
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_results_training_mode_{timestamp}.xlsx"
        
        if not output_path.endswith('.xlsx'):
            output_path = output_path + '.xlsx'
        
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_excel(output_path, index=False)
        print(f"✅ 结果已保存到: {output_path}")
        
        return results_df

    
    def print_statistics(self, results_df: pd.DataFrame):
        """打印统计信息"""
        print()
        print("=" * 100)
        print("📈 测试统计信息（训练模式 - 多样本）")
        print("=" * 100)
        
        total = len(results_df)
        
        if total == 0:
            print("⚠️  没有成功测试的样本")
            print("=" * 100)
            return
        
        # 统计准确率
        avg_accuracy_mixed = results_df['accuracy_mixed'].mean() * 100
        avg_accuracy_final = results_df['accuracy_final'].mean() * 100
        
        total_students = results_df['num_students'].sum()
        total_correct_mixed = results_df['correct_count_mixed'].sum()
        total_correct_final = results_df['correct_count_final'].sum()
        
        print(f"总样本数: {total}")
        print(f"总Student响应数: {total_students}")
        print()
        print(f"准确率 (混合分数):")
        print(f"  平均准确率: {avg_accuracy_mixed:.2f}%")
        print(f"  总体准确率: {total_correct_mixed}/{total_students} ({total_correct_mixed/total_students*100:.2f}%)")
        print()
        print(f"准确率 (最终分数):")
        print(f"  平均准确率: {avg_accuracy_final:.2f}%")
        print(f"  总体准确率: {total_correct_final}/{total_students} ({total_correct_final/total_students*100:.2f}%)")
        print()
        
        # 分数统计
        print("Teacher模型统计:")
        print(f"  原始分数(sum):    均值={results_df['teacher_score_raw'].mean():.4f}, 标准差={results_df['teacher_score_raw'].std():.4f}")
        print(f"  归一化分数(mean): 均值={results_df['teacher_score_norm'].mean():.4f}, 标准差={results_df['teacher_score_norm'].std():.4f}")
        print(f"  混合分数(70/30):  均值={results_df['teacher_score_mixed'].mean():.4f}, 标准差={results_df['teacher_score_mixed'].std():.4f}")
        print(f"  最终分数(+norm):  均值={results_df['teacher_score_final'].mean():.4f}, 标准差={results_df['teacher_score_final'].std():.4f}")
        print(f"  平均长度: {results_df['teacher_length'].mean():.2f}")
        print()
        
        print("Student模型统计 (平均):")
        print(f"  原始分数(sum):    均值={results_df['student_score_raw_mean'].mean():.4f}, 标准差={results_df['student_score_raw_mean'].std():.4f}")
        print(f"  归一化分数(mean): 均值={results_df['student_score_norm_mean'].mean():.4f}, 标准差={results_df['student_score_norm_mean'].std():.4f}")
        print(f"  混合分数(70/30):  均值={results_df['student_score_mixed_mean'].mean():.4f}, 标准差={results_df['student_score_mixed_mean'].std():.4f}")
        print(f"  最终分数(+norm):  均值={results_df['student_score_final_mean'].mean():.4f}, 标准差={results_df['student_score_final_mean'].std():.4f}")
        print(f"  平均长度: {results_df['student_length_mean'].mean():.2f}")
        print()
        
        print("分数差异统计 (平均):")
        print(f"  混合分差:   均值={results_df['score_diff_mixed_mean'].mean():.4f}, 标准差={results_df['score_diff_mixed_mean'].std():.4f}")
        print(f"  最终分差:   均值={results_df['score_diff_final_mean'].mean():.4f}, 标准差={results_df['score_diff_final_mean'].std():.4f}")
        print(f"  缩放分差:   均值={results_df['score_diff_scaled_mean'].mean():.4f}, 标准差={results_df['score_diff_scaled_mean'].std():.4f}")
        print()
        
        # 诊断信息
        print("🔍 诊断信息:")
        
        # 检查分数关系
        teacher_mixed_mean = results_df['teacher_score_mixed'].mean()
        student_mixed_mean = results_df['student_score_mixed_mean'].mean()
        
        if student_mixed_mean > teacher_mixed_mean:
            print(f"  ⚠️  Student平均混合分数 ({student_mixed_mean:.2f}) > Teacher平均混合分数 ({teacher_mixed_mean:.2f})")
            print(f"     这说明Student在某些样本上表现优于Teacher")
        else:
            print(f"  ✅ Teacher平均混合分数 ({teacher_mixed_mean:.2f}) > Student平均混合分数 ({student_mixed_mean:.2f})")
        
        # Batch normalization效果
        if results_df['use_batch_norm'].iloc[0]:
            avg_batch_size = results_df['batch_size'].mean()
            print(f"\n  ✅ Batch Normalization已启用")
            print(f"     平均batch size: {avg_batch_size:.1f} (1 teacher + {avg_batch_size-1:.0f} students)")
            print(f"     这与训练时的batch处理方式一致")
        
        print()
        
        print("配置信息:")
        print(f"  每个prompt的Student样本数: {self.training_config.get('num_student_samples', 8)}")
        print(f"  Raw weight: {self.training_config.get('raw_weight', 0.7)}")
        print(f"  Norm weight: {self.training_config.get('norm_weight', 0.3)}")
        print(f"  Use batch norm: {self.training_config.get('use_batch_norm', True)}")
        print(f"  Temperature: {self.training_config.get('temperature', 5.0)}")
        print(f"  Adaptive temperature: {self.training_config.get('adaptive_temperature', False)}")
        print()
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="测试训练后的Critic和Student模型（训练模式）")
    parser.add_argument("--data_path", type=str, default=None,
                       help="测试数据集路径")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="测试样本数量")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出Excel路径")
    
    args = parser.parse_args()
    
    # 使用配置
    data_path = args.data_path or TEST_CONFIG["data_path"]
    num_samples = args.num_samples if args.num_samples is not None else TEST_CONFIG["num_samples"]
    
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = TEST_CONFIG["output_dir"]
        output_filename = TEST_CONFIG.get("output_filename")
        
        if output_filename:
            output_path = f"{output_dir}/{output_filename}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/test_results_training_mode_{timestamp}.xlsx"
    
    critic_config = API_CONFIGS["critic_model"]
    student_config = API_CONFIGS["student_model"]
    
    print("=" * 100)
    print("📋 测试配置信息（训练模式）")
    print("=" * 100)
    print(f"数据集路径: {data_path}")
    print(f"测试样本数: {num_samples if num_samples else '全部'}")
    print(f"输出路径: {output_path}")
    print()
    print(f"Critic模型: {critic_config['name']} (本地) @ {critic_config['model_path']}")
    print(f"Student模型: {student_config['name']} @ {student_config['url']}")
    print()
    print("训练模式配置:")
    print(f"  Raw weight: {TEST_CONFIG['raw_weight']}")
    print(f"  Norm weight: {TEST_CONFIG['norm_weight']}")
    print(f"  Use batch norm: {TEST_CONFIG['use_batch_norm']}")
    print(f"  Temperature: {TEST_CONFIG['temperature']}")
    print(f"  Adaptive temperature: {TEST_CONFIG['adaptive_temperature']}")
    print("=" * 100)
    print()
    
    # 创建测试器
    tester = ModelTesterTrainingMode(
        critic_config=critic_config,
        student_config=student_config,
        training_config=TEST_CONFIG
    )
    
    # 运行测试
    results_df = tester.test_dataset(
        data_path=data_path,
        num_samples=num_samples,
        output_path=output_path,
        random_sample=TEST_CONFIG.get("random_sample", False),
        random_seed=TEST_CONFIG.get("random_seed", 42)
    )
    
    print("✅ 测试完成！")


if __name__ == "__main__":
    main()
