"""
测试训练后的Critic和Student模型
用于评估模型打分能力和训练效果
"""
import requests
import pandas as pd
import argparse
from typing import Dict, Optional
from datetime import datetime
from tqdm import tqdm
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== 配置区域 ====================
# 在这里修改你的模型配置和测试参数

# 测试参数配置
TEST_CONFIG = {
    "data_path": "/home/jovyan/JQ/gad_gspo_B300/data/trainning_dataset/subject_1-29/merged/merge-1-29.parquet",  # 测试数据集路径
    "num_samples": 10,  # 测试样本数量（None表示全部）
    "random_sample": True,  # 是否随机抽样（True=随机，False=取前N个）
    "random_seed": 42,  # 随机种子（用于可复现的随机抽样）
    "output_dir": "/home/jovyan/JQ/gad_gspo_B300/outputs",  # 输出目录
    "output_filename": None  # 输出文件名（None表示自动生成带时间戳的文件名）
}

# 模型配置
API_CONFIGS = {
    "critic_model": {
        "name": "critic-model",
        "type": "local",  # "local" 表示本地加载，"api" 表示API调用
        "model_path": "/home/jovyan/JQ/gad_gspo_B300/models/opd-v9-1-29-fsdp2/global_step_500/critic_merged",  # 本地模型路径
        "device": "cuda:4",  # 使用的设备
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "force_trl": True,  # 强制使用trl加载（跳过TokenClassification尝试）
        "use_simple_length": False  # 使用简单的长度计算（split()而不是tokenizer）
    },
    "student_model": {
        "name": "student-model",
        "type": "api",  # API调用
        "url": "http://10.72.1.39:8008/v1/chat/completions",
        "api_key": "sk-xxxx",
        "model_name": "opd-v9-500",
        "temperature": 0.6,
        "repetition_penalty": 1.2
    }
}
# ================================================


class ModelTester:
    def __init__(self, critic_config: Dict, student_config: Dict):
        """
        初始化模型测试器
        
        Args:
            critic_config: Critic模型配置字典
            student_config: Student模型配置字典
        """
        self.critic_config = critic_config
        self.student_config = student_config
        
        # 如果Critic是本地模型，加载它
        self.critic_model = None
        self.critic_tokenizer = None
        if critic_config.get("type") == "local":
            print(f"🔄 加载本地Critic模型: {critic_config['model_path']}")
            self.load_local_critic(critic_config)
            print(f"✅ Critic模型加载完成")
    
    def load_local_critic(self, config: Dict):
        """
        加载本地Critic模型
        
        Args:
            config: Critic模型配置
        """
        device = config.get("device", "cuda:0")
        model_path = config["model_path"]
        force_trl = config.get("force_trl", False)
        
        print(f"🔄 加载tokenizer...")
        # 加载tokenizer
        self.critic_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer加载完成")
        
        # 如果强制使用trl，直接跳过TokenClassification尝试
        if force_trl:
            print(f"⚡ 配置强制使用trl，跳过TokenClassification尝试")
            self._load_with_trl(model_path, device)
            return
        
        # 🔧 关键：按照训练代码的逻辑加载模型
        # 首先尝试 AutoModelForTokenClassification，失败则使用 trl
        from transformers import AutoModelForTokenClassification, AutoConfig
        
        # 先检查模型配置
        print(f"🔍 检查模型配置...")
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  架构: {model_config.architectures}")
        print(f"  num_labels: {getattr(model_config, 'num_labels', 'N/A')}")
        
        # 判断是否是TokenClassification模型
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
        
        # 如果不是TokenClassification，使用trl
        print(f"🔄 模型不是TokenClassification类型，使用trl加载...")
        self._load_with_trl(model_path, device)
    
    def _load_with_trl(self, model_path: str, device: str):
        """使用trl加载模型"""
        try:
            from trl import AutoModelForCausalLMWithValueHead
            from transformers import AutoModelForCausalLM
            
            print(f"  加载基础CausalLM模型...")
            # 先加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device
            )
            
            print(f"  包装为ValueHead模型...")
            # 包装为ValueHead模型
            self.critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
            print(f"✅ 成功使用 trl.AutoModelForCausalLMWithValueHead 加载")
            self.critic_model.eval()
        except Exception as e:
            print(f"❌ trl加载失败: {e}")
            raise RuntimeError(f"无法加载Critic模型: {e}")
    
    def call_generation_model(self, config: Dict, prompt: str, max_tokens: int = 512) -> Dict:
        """
        调用生成模型API
        
        Args:
            config: 模型配置字典
            prompt: 输入提示
            max_tokens: 最大生成token数
            
        Returns:
            包含生成文本和token数的字典
        """
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
    
    def call_critic_model(self, config: Dict, prompt: str, response: str) -> float:
        """
        调用Critic模型进行打分
        
        Args:
            config: Critic模型配置字典
            prompt: 原始提示
            response: 模型响应
            
        Returns:
            分数
        """
        # 判断是本地模型还是API
        if config.get("type") == "local":
            return self.call_local_critic(prompt, response)
        else:
            return self.call_api_critic(config, prompt, response)
    
    def call_local_critic(self, prompt: str, response: str) -> float:
        """
        调用本地Critic模型进行打分
        
        ⚠️ 重要：此方法必须与训练时的分数计算逻辑完全一致！
        
        关键修复（2026-02-03）：
        1. 值提取顺序：先提取response部分，再squeeze（与训练代码dp_critic.py一致）
           - 训练代码：values[:, -response_length:].squeeze(-1)
           - 之前错误：先squeeze整个序列，再提取response（导致维度错位）
        2. EOS token排除：显式排除EOS token后计算平均值
        3. 平均值机制：对response所有token（排除EOS）的values取平均
        
        Args:
            prompt: 原始提示
            response: 模型响应
            
        Returns:
            分数（与训练时计算方式一致）
        """
        try:
            # 构建输入 - 使用与训练时相同的格式
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # 使用tokenizer的chat template
            if hasattr(self.critic_tokenizer, 'apply_chat_template'):
                input_text = self.critic_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # 简单拼接（如果没有chat template）
                input_text = f"User: {prompt}\nAssistant: {response}"
            
            # Tokenize
            inputs = self.critic_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # 🔧 获取模型设备（兼容不同模型类型）
            if hasattr(self.critic_model, 'device'):
                device = self.critic_model.device
            elif hasattr(self.critic_model, 'pretrained_model'):
                # trl模型的设备在pretrained_model中
                device = next(self.critic_model.pretrained_model.parameters()).device
            else:
                # 从第一个参数获取设备
                device = next(self.critic_model.parameters()).device
            
            inputs = inputs.to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # 前向传播获取所有token的values
            with torch.no_grad():
                outputs = self.critic_model(**inputs, use_cache=False)
                
                # 🔧 根据模型类型获取values - 与训练代码完全一致
                # 方式1: AutoModelForTokenClassification (有logits属性)
                if hasattr(outputs, 'logits'):
                    values = outputs.logits
                    # 🔧 关键：先提取response部分，再squeeze（与训练代码一致）
                    # 重新tokenize response以获取其长度
                    response_tokens = self.critic_tokenizer(
                        response,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )
                    response_length = response_tokens['input_ids'].size(1)
                    
                    # 先提取response部分
                    values = values[:, -response_length:]  # (batch, response_length, 1) or (batch, response_length)
                    # 再squeeze
                    if values.dim() == 3:
                        values = values.squeeze(-1)  # (batch, response_length)
                
                # 方式2: trl.AutoModelForCausalLMWithValueHead (返回tuple)
                elif hasattr(self.critic_model, 'v_head') or isinstance(outputs, tuple):
                    if isinstance(outputs, tuple) and len(outputs) > 2:
                        values = outputs[2]  # value head的输出 (batch, seq_len, 1) or (batch, seq_len)
                        
                        # 🔧 关键：与训练代码完全一致的处理顺序
                        # 1. 先提取response部分
                        # 2. 再squeeze最后一维
                        
                        # 重新tokenize response以获取其长度
                        response_tokens = self.critic_tokenizer(
                            response,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )
                        response_length = response_tokens['input_ids'].size(1)
                        
                        # 先提取response部分
                        values = values[:, -response_length:]  # (batch, response_length, 1) or (batch, response_length)
                        
                        # 🔧 调试：打印values的形状（在squeeze之前）
                        if not hasattr(self, '_values_shape_printed'):
                            print(f"\n🔍 Values调试信息:")
                            print(f"  提取response后 values shape: {values.shape}")
                            print(f"  Response length: {response_length}")
                            print(f"  原始values dtype: {values.dtype}")
                            print(f"  原始values范围: [{values.min().item():.4f}, {values.max().item():.4f}]")
                            print(f"  原始values均值: {values.mean().item():.4f}")
                            self._values_shape_printed = True
                        
                        # 再squeeze最后一维
                        if values.dim() == 3:
                            values = values.squeeze(-1)  # (batch, response_length)
                    else:
                        raise ValueError("模型返回tuple但格式不正确")
                else:
                    raise ValueError(f"无法识别的模型输出格式: {type(outputs)}")
            
            # 🔧 关键：与训练代码保持一致的分数计算
            # 1. values已经是response部分了（在上面提取过）
            # 2. 排除EOS token
            # 3. 计算平均值
            
            # values已经是response部分，直接使用
            response_values = values  # (1, response_length)
            response_mask = attention_mask[:, -response_length:]  # (1, response_length)
            
            # 获取response的token IDs（用于识别EOS）
            response_ids = input_ids[:, -response_length:]
            
            # 获取EOS token ID
            eos_token_id = self.critic_tokenizer.eos_token_id
            
            # 🔧 关键修复：排除EOS token
            is_eos = (response_ids == eos_token_id)
            response_mask_no_eos = response_mask & (~is_eos)
            
            # 计算平均值（与训练代码完全一致）
            values_sum = (response_values * response_mask_no_eos).sum(dim=-1)  # (1,)
            values_count = response_mask_no_eos.sum(dim=-1).clamp(min=1)  # (1,)
            score = (values_sum / values_count).item()  # scalar
            
            return float(score)
            
        except Exception as e:
            print(f"❌ 本地Critic模型调用失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def call_api_critic(self, config: Dict, prompt: str, response: str) -> float:
        """
        通过API调用Critic模型进行打分
        
        Args:
            config: Critic模型配置字典
            prompt: 原始提示
            response: 模型响应
            
        Returns:
            分数
        """
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            payload = {
                "model": config["model_name"],
                "messages": messages,
                "max_tokens": 1,
                "temperature": config.get("temperature", 0.0),
                "logprobs": True,
                "top_logprobs": 1
            }
            
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            response_obj = requests.post(
                config["url"],
                json=payload,
                headers=headers,
                timeout=30
            )
            response_obj.raise_for_status()
            
            result = response_obj.json()
            
            # 尝试提取分数
            if 'score' in result:
                score = float(result['score'])
            elif 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    logprobs = choice['logprobs']
                    if 'content' in logprobs and len(logprobs['content']) > 0:
                        score = logprobs['content'][0].get('logprob', 0.0)
                    else:
                        score = 0.0
                elif 'message' in choice:
                    content = choice['message'].get('content', '0.0')
                    try:
                        score = float(content.strip())
                    except:
                        score = 0.0
                else:
                    score = 0.0
            else:
                score = 0.0
            
            return score
        except Exception as e:
            print(f"❌ Critic API调用失败 [{config['name']}]: {e}")
            return 0.0
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        格式化提示词
        
        Args:
            instruction: 指令
            input_text: 输入文本
            
        Returns:
            格式化后的提示
        """
        if input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction
    
    def test_single_sample(self, prompt: str, teacher_response: Optional[str] = None) -> Dict:
        """
        测试单个样本
        
        Args:
            prompt: 输入提示
            teacher_response: 教师响应（可选）
            
        Returns:
            测试结果字典
        """
        print("📝 Prompt:")
        print(f"  {prompt}")
        print()
        
        # 生成Student响应
        student_result = self.call_generation_model(self.student_config, prompt)
        student_text = student_result['text']
        student_length = student_result['length']
        
        # 获取Critic对Student的打分
        student_score = self.call_critic_model(self.critic_config, prompt, student_text)
        
        print("🎓 Student Response:")
        print("-" * 100)
        print(f"  Score: {student_score:7.4f} | Length: {student_length:3d}")
        print(f"  Text: {student_text}")
        print()
        
        result = {
            'prompt': prompt,
            'student_response': student_text,
            'student_score': student_score,
            'student_length': student_length
        }
        
        # 使用数据集中的teacher响应
        teacher_text = teacher_response
        
        # 🔧 修复：使用tokenizer计算准确的token长度（或使用简单计算）
        if self.critic_config.get("use_simple_length", False):
            # 简单计算：按字符数（中文）或单词数（英文）
            teacher_length = len(teacher_text) if any('\u4e00' <= c <= '\u9fff' for c in teacher_text[:100]) else len(teacher_text.split())
        else:
            # 精确计算：使用tokenizer
            teacher_tokens = self.critic_tokenizer(
                teacher_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            teacher_length = teacher_tokens['input_ids'].size(1)
        
        # 获取Critic对Teacher的打分
        teacher_score = self.call_critic_model(self.critic_config, prompt, teacher_text)
        
        print("👨‍🏫 Teacher Response:")
        print("-" * 100)
        print(f"  Score: {teacher_score:7.4f} | Length: {teacher_length:3d}")
        print(f"  Text: {teacher_text}")
        print()
        
        # 计算分数差异
        score_diff = teacher_score - student_score
        is_correct = teacher_score > student_score
        
        print("📊 分数对比:")
        print(f"  Teacher - Student = {score_diff:7.4f}")
        print(f"  Teacher > Student: {'✅ 正确' if is_correct else '❌ 错误'}")
        print()
        print("=" * 100)
        print()
        
        result.update({
            'teacher_response': teacher_text,
            'teacher_score': teacher_score,
            'teacher_length': teacher_length,
            'score_diff': score_diff,
            'is_correct': is_correct
        })
        
        return result
    
    def test_dataset(self, data_path: str, num_samples: Optional[int] = None, 
                    output_path: Optional[str] = None, random_sample: bool = False,
                    random_seed: int = 42) -> pd.DataFrame:
        """
        测试数据集
        
        Args:
            data_path: 数据集路径（支持.parquet或.xlsx）
            num_samples: 测试样本数量（None表示全部）
            output_path: 输出Excel路径
            random_sample: 是否随机抽样
            random_seed: 随机种子
            
        Returns:
            测试结果DataFrame
        """
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
                # 随机抽样
                df = df.sample(n=min(num_samples, len(df)), random_state=random_seed)
                print(f"📊 随机抽样 {len(df)} 个样本（种子={random_seed}）...")
            else:
                # 取前N个
                df = df.head(num_samples)
                print(f"📊 取前 {len(df)} 个样本...")
        else:
            print(f"📊 测试全部 {len(df)} 个样本...")
        print()
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="测试进度"):
            try:
                # 解析content字段 - content是一个数组，包含对话内容
                if 'content' in row:
                    content = row['content']
                    if isinstance(content, (list, tuple)) and len(content) > 0:
                        # 获取第一个元素的content字段作为prompt
                        prompt = content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
                    else:
                        prompt = str(content)
                elif 'instruction' in row and 'input' in row:
                    prompt = self.format_prompt(row['instruction'], row.get('input', ''))
                elif 'prompt' in row:
                    prompt = row['prompt']
                elif 'question' in row:
                    prompt = row['question']
                else:
                    print(f"⚠️  跳过第 {idx} 行：无法识别的数据格式")
                    continue
                
                # 获取teacher_response
                teacher_response = row.get('teacher_response', '')
                
                if not prompt or not teacher_response:
                    print(f"⚠️  跳过第 {idx} 行：缺少prompt或teacher_response")
                    continue
                
                result = self.test_single_sample(prompt, teacher_response)
                result['sample_id'] = idx
                result['data_id'] = row.get('id', idx)
                results.append(result)
                
                # 避免请求过快
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ 测试第 {idx} 个样本时出错: {e}")
                continue
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 计算统计信息
        self.print_statistics(results_df)
        
        # 保存结果
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_results_{timestamp}.xlsx"
        
        # 确保输出路径有.xlsx扩展名
        if not output_path.endswith('.xlsx'):
            output_path = output_path + '.xlsx'
        
        # 确保输出目录存在
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_excel(output_path, index=False)
        print(f"✅ 结果已保存到: {output_path}")
        
        return results_df
    
    def print_statistics(self, results_df: pd.DataFrame):
        """
        打印统计信息
        
        Args:
            results_df: 测试结果DataFrame
        """
        print()
        print("=" * 100)
        print("📈 测试统计信息")
        print("=" * 100)
        
        total = len(results_df)
        
        if total == 0:
            print("⚠️  没有成功测试的样本")
            print("=" * 100)
            return
        
        correct = results_df['is_correct'].sum()
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"总样本数: {total}")
        print(f"正确判断数: {correct}")
        print(f"准确率: {accuracy:.2f}%")
        print()
        
        print("Student模型统计:")
        print(f"  平均分数: {results_df['student_score'].mean():.4f}")
        print(f"  分数标准差: {results_df['student_score'].std():.4f}")
        print(f"  平均长度: {results_df['student_length'].mean():.2f}")
        print()
        
        print("Teacher模型统计:")
        print(f"  平均分数: {results_df['teacher_score'].mean():.4f}")
        print(f"  分数标准差: {results_df['teacher_score'].std():.4f}")
        print(f"  平均长度: {results_df['teacher_length'].mean():.2f}")
        print()
        
        print("分数差异统计:")
        print(f"  平均差异: {results_df['score_diff'].mean():.4f}")
        print(f"  差异标准差: {results_df['score_diff'].std():.4f}")
        print(f"  最大差异: {results_df['score_diff'].max():.4f}")
        print(f"  最小差异: {results_df['score_diff'].min():.4f}")
        print()
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="测试训练后的Critic和Student模型")
    parser.add_argument("--data_path", type=str, default=None,
                       help="测试数据集路径（默认使用配置中的路径）")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="测试样本数量（默认使用配置中的数量）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出Excel路径（默认使用配置中的路径）")
    
    args = parser.parse_args()
    
    # 使用命令行参数或配置文件中的值
    data_path = args.data_path or TEST_CONFIG["data_path"]
    num_samples = args.num_samples if args.num_samples is not None else TEST_CONFIG["num_samples"]
    
    # 处理输出路径
    if args.output_path:
        output_path = args.output_path
    else:
        # 使用配置中的输出目录和文件名
        output_dir = TEST_CONFIG["output_dir"]
        output_filename = TEST_CONFIG.get("output_filename")
        
        if output_filename:
            output_path = f"{output_dir}/{output_filename}"
        else:
            # 自动生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/test_results_{timestamp}.xlsx"
    
    # 使用脚本顶部的配置
    critic_config = API_CONFIGS["critic_model"]
    student_config = API_CONFIGS["student_model"]
    
    print("=" * 100)
    print("📋 测试配置信息")
    print("=" * 100)
    print(f"数据集路径: {data_path}")
    print(f"测试样本数: {num_samples if num_samples else '全部'}")
    print(f"输出路径: {output_path}")
    print()
    if critic_config.get("type") == "local":
        print(f"Critic模型: {critic_config['name']} (本地) @ {critic_config['model_path']}")
    else:
        print(f"Critic模型: {critic_config['name']} @ {critic_config['url']}")
    print(f"Student模型: {student_config['name']} @ {student_config['url']}")
    print(f"Teacher响应: 从数据集读取")
    print("=" * 100)
    print()
    
    # 创建测试器
    tester = ModelTester(
        critic_config=critic_config,
        student_config=student_config
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
