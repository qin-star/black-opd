#!/usr/bin/env python3
"""
将 core_content 训练数据转换为 GAD 训练所需的 parquet 格式
excel分为两列：prompt || teacher_response

输入：core_content_trainning_data.xlsx
输出：训练集和测试集的Excel和parquet文件（2:8比例切分）train_ratio 可调
"""

import pandas as pd
import json
import os
import random
from typing import List, Dict, Tuple


def convert_row(row: pd.Series) -> Dict:
    """
    转换单行数据
    
    Args:
        row: Excel 中的一行数据
        
    Returns:
        转换后的字典，包含 content 和 teacher_response
    """
    # 直接从第一列获取 prompt
    prompt = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
    
    # 直接从第二列获取 teacher_response
    teacher_response = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
    
    # 构造 content（消息列表格式）
    content = [
        {
            "content": prompt,
            "role": "user"
        }
    ]
    
    return {
        "content": content,
        "teacher_response": teacher_response
    }


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.9, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    随机切分数据集为训练集和测试集
    
    Args:
        df: 原始数据框
        train_ratio: 训练集比例，默认0.8（即2:8的测试集:训练集比例）
        random_seed: 随机种子，确保可重现性
        
    Returns:
        训练集和测试集的数据框元组 (train_df, test_df)
    """
    # 设置随机种子确保可重现性
    random.seed(random_seed)
    
    # 随机打乱数据
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 计算切分点
    split_idx = int(len(shuffled_df) * train_ratio)
    
    # 切分数据
    train_df = shuffled_df.iloc[:split_idx].reset_index(drop=True)
    test_df = shuffled_df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def save_dataset(df: pd.DataFrame, output_dir: str, base_name: str, dataset_type: str) -> Tuple[str, str]:
    """
    保存数据集为Excel和Parquet格式
    
    Args:
        df: 要保存的数据框
        output_dir: 输出目录
        base_name: 基础文件名
        dataset_type: 数据集类型（train/test）
        
    Returns:
        Excel文件和Parquet文件的路径元组
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造文件名
    excel_filename = f"{base_name}_{dataset_type}.xlsx"
    parquet_filename = f"{base_name}_{dataset_type}.parquet"
    
    excel_path = os.path.join(output_dir, excel_filename)
    parquet_path = os.path.join(output_dir, parquet_filename)
    
    # 保存Excel文件（保存原始格式）
    df.to_excel(excel_path, index=False)
    
    # 转换并保存为Parquet格式
    converted_data = []
    for idx, row in df.iterrows():
        try:
            converted_row = convert_row(row)
            converted_data.append(converted_row)
        except Exception as e:
            print(f"   ⚠️  第 {idx} 行转换失败: {e}")
            continue
    
    # 创建转换后的DataFrame
    result_df = pd.DataFrame(converted_data)
    
    # 添加id列
    result_df['id'] = [f"{base_name}_{dataset_type}_{i:06d}" for i in range(len(result_df))]
    
    # 调整列顺序
    result_df = result_df[['id', 'content', 'teacher_response']]
    
    # 保存Parquet文件
    result_df.to_parquet(parquet_path, index=False)
    
    return excel_path, parquet_path


def main():
    """主函数"""
    print("=" * 60)
    print("Core Content Excel 数据转换为 GAD Parquet 格式")
    print("=" * 60)
    
    print("=" * 60)
    print("Core Content Excel 数据转换为 GAD Parquet 格式（支持训练集/测试集切分）")
    print("=" * 60)
    
    # 1. 获取输入文件路径
    excel_path = "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/semantic_understanding/semantic_understanding_1224.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ 错误：文件不存在: {excel_path}")
        return
    
    # 2. 确定输出目录（与输入Excel同目录下的processed文件夹）
    excel_dir = os.path.dirname(excel_path)
    output_dir = os.path.join(excel_dir, "processed")
    
    # 3. 获取基础文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    
    print(f"\n📖 读取 Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"   ✅ 读取成功，共 {len(df)} 行")
    print(f"   列名: {df.columns.tolist()}")
    print(f"   输出目录: {output_dir}")
    
    # 4. 切分数据集
    print(f"\n🔄 切分数据集（训练集:测试集 = 9:1）...")
    train_df, test_df = split_dataset(df, train_ratio=0.99, random_seed=42)
    print(f"   ✅ 切分完成")
    print(f"   训练集: {len(train_df)} 行 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   测试集: {len(test_df)} 行 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 5. 保存训练集
    print(f"\n💾 保存训练集...")
    train_excel_path, train_parquet_path = save_dataset(train_df, output_dir, base_name, "train")
    print(f"   ✅ 训练集保存成功")
    print(f"   Excel: {train_excel_path}")
    print(f"   Parquet: {train_parquet_path}")
    
    # 6. 保存测试集
    print(f"\n💾 保存测试集...")
    test_excel_path, test_parquet_path = save_dataset(test_df, output_dir, base_name, "test")
    print(f"   ✅ 测试集保存成功")
    print(f"   Excel: {test_excel_path}")
    print(f"   Parquet: {test_parquet_path}")
    
    # 7. 验证输出文件
    print(f"\n🔍 验证输出文件...")
    
    # 验证训练集Parquet
    try:
        verify_train = pd.read_parquet(train_parquet_path)
        print(f"   训练集Parquet: {len(verify_train)} 行，列: {verify_train.columns.tolist()}")
    except Exception as e:
        print(f"   ⚠️  训练集Parquet验证失败: {e}")
    
    # 验证测试集Parquet
    try:
        verify_test = pd.read_parquet(test_parquet_path)
        print(f"   测试集Parquet: {len(verify_test)} 行，列: {verify_test.columns.tolist()}")
    except Exception as e:
        print(f"   ⚠️  测试集Parquet验证失败: {e}")
    
    # 8. 显示示例数据
    if len(verify_train) > 0:
        print(f"\n📝 训练集第一行示例:")
        first_row = verify_train.iloc[0]
        print(f"   ID: {first_row['id']}")
        print(f"   Content (前 200 字符): {str(first_row['content'])[:200]}...")
        print(f"   Teacher Response (前 200 字符): {str(first_row['teacher_response'])[:200]}...")
    
    if len(verify_test) > 0:
        print(f"\n📝 测试集第一行示例:")
        first_row = verify_test.iloc[0]
        print(f"   ID: {first_row['id']}")
        print(f"   Content (前 200 字符): {str(first_row['content'])[:200]}...")
        print(f"   Teacher Response (前 200 字符): {str(first_row['teacher_response'])[:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ 数据集切分和转换完成！")
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    print(f"  - {os.path.basename(train_excel_path)} (训练集Excel)")
    print(f"  - {os.path.basename(train_parquet_path)} (训练集Parquet)")
    print(f"  - {os.path.basename(test_excel_path)} (测试集Excel)")
    print(f"  - {os.path.basename(test_parquet_path)} (测试集Parquet)")
    print("=" * 60)


if __name__ == "__main__":
    main()