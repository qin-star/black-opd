#!/usr/bin/env python3
"""
从 GenRM 评分结果 Excel 中提取 score_result 列的 ranking 字段

输入：包含 score_result 列的 Excel 文件（JSON 格式字符串）
输出：在原 Excel 基础上新增 ranking 列

score_result 示例格式：
{
    "response_1_analysis": "...",
    "response_2_analysis": "...",
    "score_1": 4,
    "score_2": 5,
    "ranking": 5
}
"""

import pandas as pd
import json
import re
import os
from typing import Optional, Union


def extract_ranking(score_result: str) -> Optional[int]:
    """
    从 score_result JSON 字符串中提取 ranking 值
    
    Args:
        score_result: JSON 格式的评分结果字符串
        
    Returns:
        ranking 值（1-6），解析失败返回 None
    """
    if pd.isna(score_result) or not score_result:
        return None
    
    try:
        # 尝试直接解析 JSON
        data = json.loads(score_result)
        ranking = data.get('ranking')
        if ranking is not None:
            return int(ranking)
    except json.JSONDecodeError:
        pass
    
    # 如果 JSON 解析失败，尝试用正则表达式提取
    try:
        # 匹配 "ranking": 数字 或 "ranking":数字
        match = re.search(r'"ranking"\s*:\s*(\d+)', str(score_result))
        if match:
            return int(match.group(1))
    except Exception:
        pass
    
    return None


def extract_all_fields(score_result: str) -> dict:
    """
    从 score_result JSON 字符串中提取所有评分字段
    
    Args:
        score_result: JSON 格式的评分结果字符串
        
    Returns:
        包含 score_1, score_2, ranking 的字典
    """
    result = {
        'score_1': None,
        'score_2': None,
        'ranking': None,
        'response_1_analysis': None,
        'response_2_analysis': None
    }
    
    if pd.isna(score_result) or not score_result:
        return result
    
    try:
        data = json.loads(score_result)
        result['score_1'] = data.get('score_1')
        result['score_2'] = data.get('score_2')
        result['ranking'] = data.get('ranking')
        result['response_1_analysis'] = data.get('response_1_analysis')
        result['response_2_analysis'] = data.get('response_2_analysis')
    except json.JSONDecodeError:
        # 使用正则表达式作为备选方案
        for field in ['score_1', 'score_2', 'ranking']:
            match = re.search(rf'"{field}"\s*:\s*(\d+)', str(score_result))
            if match:
                result[field] = int(match.group(1))
    
    return result


def process_excel(input_path: str, output_path: str = None, extract_all: bool = False) -> pd.DataFrame:
    """
    处理 Excel 文件，提取 ranking 字段
    
    Args:
        input_path: 输入 Excel 文件路径
        output_path: 输出 Excel 文件路径（默认覆盖原文件）
        extract_all: 是否提取所有字段（score_1, score_2, ranking）
        
    Returns:
        处理后的 DataFrame
    """
    print(f"📖 读取 Excel: {input_path}")
    df = pd.read_excel(input_path)
    print(f"   ✅ 读取成功，共 {len(df)} 行")
    print(f"   列名: {df.columns.tolist()}")
    
    # 检查是否存在 score_result 列
    if 'score_result' not in df.columns:
        print("❌ 错误：Excel 中不存在 'score_result' 列")
        print(f"   可用列: {df.columns.tolist()}")
        return df
    
    print(f"\n🔄 提取 ranking 字段...")
    
    if extract_all:
        # 提取所有字段
        extracted = df['score_result'].apply(extract_all_fields)
        extracted_df = pd.DataFrame(extracted.tolist())
        
        # 添加新列
        for col in ['score_1', 'score_2', 'ranking']:
            if col not in df.columns:
                df[col] = extracted_df[col]
            else:
                # 如果列已存在，创建新列名
                df[f'{col}_extracted'] = extracted_df[col]
    else:
        # 只提取 ranking
        if 'ranking' not in df.columns:
            df['ranking'] = df['score_result'].apply(extract_ranking)
        else:
            df['ranking_extracted'] = df['score_result'].apply(extract_ranking)
    
    # 统计 ranking 分布
    ranking_col = 'ranking' if 'ranking' in df.columns else 'ranking_extracted'
    print(f"\n📊 Ranking 分布统计:")
    ranking_counts = df[ranking_col].value_counts().sort_index()
    for rank, count in ranking_counts.items():
        print(f"   Ranking {rank}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    null_count = df[ranking_col].isna().sum()
    if null_count > 0:
        print(f"   ⚠️  解析失败: {null_count} 条 ({null_count/len(df)*100:.1f}%)")
    
    # 保存结果
    if output_path is None:
        output_path = input_path
    
    print(f"\n💾 保存结果到: {output_path}")
    df.to_excel(output_path, index=False)
    print(f"   ✅ 保存成功")
    
    return df


def main():
    """主函数"""
    print("=" * 60)
    print("GenRM 评分结果 Ranking 提取工具")
    print("=" * 60)
    
    # 输入文件路径
    input_path = r"E:\LLM-trainning\gad_gspo_b300\data\genrm_scores_20260106_223755.xlsx"
    
    # 输出文件路径（可以设置为 None 覆盖原文件，或指定新路径）
    output_path = r"E:\LLM-trainning\gad_gspo_b300\data\genrm_scores_20260106_223755_with_ranking.xlsx"
    
    if not os.path.exists(input_path):
        print(f"❌ 错误：文件不存在: {input_path}")
        return
    
    # 处理 Excel，提取 ranking（设置 extract_all=True 可提取所有字段）
    df = process_excel(input_path, output_path, extract_all=False)
    
    # 显示示例
    if len(df) > 0:
        print(f"\n📝 前 5 行示例:")
        ranking_col = 'ranking' if 'ranking' in df.columns else 'ranking_extracted'
        print(df[[ranking_col]].head())
    
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
