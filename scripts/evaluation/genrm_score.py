"""
GenRM模型评分脚本
用于对验证数据集进行评分
"""
import pandas as pd
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import os
from datetime import datetime

def load_dataset(file_path):
    """加载Excel数据集"""
    print(f"正在加载数据集: {file_path}")
    df = pd.read_excel(file_path)
    print(f"数据集列名: {df.columns.tolist()}")
    print(f"数据集大小: {len(df)} 条")
    return df

def build_messages(row, prompt_col, response1_col, response2_col, history_col=None):
    """
    构建消息列表
    
    Args:
        row: DataFrame的一行数据
        prompt_col: 用户问题列名
        response1_col: 第一个回答列名
        response2_col: 第二个回答列名
        history_col: 历史对话列名（可选）
    """
    messages = []
    
    # 如果有历史对话，先添加历史
    if history_col and pd.notna(row.get(history_col)):
        history = row[history_col]
        if isinstance(history, str):
            try:
                history = json.loads(history)
            except:
                pass
        
        if isinstance(history, list):
            for h in history:
                if isinstance(h, dict) and 'role' in h and 'content' in h:
                    messages.append(h)
    
    # 添加当前用户问题
    if pd.notna(row.get(prompt_col)):
        messages.append({
            "role": "user",
            "content": str(row[prompt_col])
        })
    
    # 添加两个候选回答
    if pd.notna(row.get(response1_col)):
        messages.append({
            "role": "response_1",
            "content": str(row[response1_col])
        })
    
    if pd.notna(row.get(response2_col)):
        messages.append({
            "role": "response_2",
            "content": str(row[response2_col])
        })
    
    return messages

def call_genrm(client, messages, model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM"):
    """调用GenRM模型进行评分"""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
            stream=False
        )
        
        output = completion.choices[0].message.content
        # 提取</think>标签后的内容
        result = output.split("</think>")[-1].strip()
        return result, output
    except Exception as e:
        print(f"调用模型出错: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='GenRM模型评分脚本')
    parser.add_argument('--data_path', type=str, 
                       default='data/GenRM_val_dataset.xlsx',
                       help='数据集路径')
    parser.add_argument('--base_url', type=str, 
                       default='http://127.0.0.1:5000/v1',
                       help='API base URL')
    parser.add_argument('--model_name', type=str,
                       default='nvidia/Qwen3-Nemotron-235B-A22B-GenRM',
                       help='模型名称')
    parser.add_argument('--prompt_col', type=str,
                       default='prompt',
                       help='问题列名')
    parser.add_argument('--response1_col', type=str,
                       default='response_1',
                       help='第一个回答列名')
    parser.add_argument('--response2_col', type=str,
                       default='response_2',
                       help='第二个回答列名')
    parser.add_argument('--history_col', type=str,
                       default=None,
                       help='历史对话列名（可选）')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/genrm_scores',
                       help='输出目录')
    parser.add_argument('--max_samples', type=int,
                       default=None,
                       help='最大处理样本数（用于测试）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化OpenAI客户端
    print(f"连接到API: {args.base_url}")
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    
    # 加载数据集
    df = load_dataset(args.data_path)
    
    # 检查列名是否存在
    required_cols = [args.prompt_col, args.response1_col, args.response2_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n错误: 数据集中缺少以下列: {missing_cols}")
        print(f"可用的列: {df.columns.tolist()}")
        print("\n请使用 --prompt_col, --response1_col, --response2_col 参数指定正确的列名")
        return
    
    # 限制样本数（用于测试）
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"限制处理样本数: {args.max_samples}")
    
    # 处理每一行数据
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="评分进度"):
        # 构建消息
        messages = build_messages(
            row, 
            args.prompt_col, 
            args.response1_col, 
            args.response2_col,
            args.history_col
        )
        
        # 调用模型
        score_result, full_output = call_genrm(client, messages, args.model_name)
        
        # 保存结果
        result = {
            'index': idx,
            'prompt': row.get(args.prompt_col),
            'response_1': row.get(args.response1_col),
            'response_2': row.get(args.response2_col),
            'score_result': score_result,
            'full_output': full_output
        }
        
        # 如果有其他列，也保存
        for col in df.columns:
            if col not in [args.prompt_col, args.response1_col, args.response2_col]:
                result[f'original_{col}'] = row.get(col)
        
        results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f'genrm_scores_{timestamp}.xlsx')
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    print(f"\n评分完成！结果已保存到: {output_file}")
    
    # 打印统计信息
    print(f"\n总样本数: {len(results)}")
    if results:
        success_count = sum(1 for r in results if r['score_result'] is not None)
        print(f"成功评分: {success_count}")
        print(f"失败数: {len(results) - success_count}")

if __name__ == "__main__":
    main()
