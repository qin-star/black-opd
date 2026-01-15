#!/usr/bin/env python3
"""
通用数据转换脚本：支持 Excel 和 JSONL 格式转换为 GAD 训练所需的 parquet 格式

支持的输入格式：
1. Excel (.xlsx, .xls): 两列格式 prompt || teacher_response
2. JSONL (.jsonl, .json): {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

输出：训练集和测试集的 parquet 文件（可配置比例切分）

"""

import pandas as pd
import json
import os
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def convert_excel_row(row: pd.Series) -> Dict:
    """
    转换 Excel 单行数据
    
    Args:
        row: Excel 中的一行数据（第一列为prompt，第二列为teacher_response）
        
    Returns:
        转换后的字典
    """
    prompt = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
    teacher_response = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
    
    content = [{"content": prompt, "role": "user"}]
    
    return {
        "content": content,
        "teacher_response": teacher_response
    }


def convert_jsonl_row(data: Dict) -> Dict:
    """
    转换 JSONL 单行数据
    
    Args:
        data: JSONL 中的一行数据，格式为 {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
    Returns:
        转换后的字典
    """
    messages = data.get("messages", [])
    
    prompt = ""
    teacher_response = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            prompt = content
        elif role == "assistant":
            teacher_response = content
    
    content = [{"content": prompt, "role": "user"}]
    
    return {
        "content": content,
        "teacher_response": teacher_response
    }


def load_excel_data(file_path: str) -> List[Dict]:
    """加载 Excel 文件数据"""
    print(f"📖 读取 Excel: {file_path}")
    df = pd.read_excel(file_path)
    print(f"   ✅ 读取成功，共 {len(df)} 行")
    print(f"   列名: {df.columns.tolist()}")
    
    converted_data = []
    for idx, row in df.iterrows():
        try:
            converted_row = convert_excel_row(row)
            converted_data.append(converted_row)
        except Exception as e:
            print(f"   ⚠️  第 {idx} 行转换失败: {e}")
            continue
    
    return converted_data


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载 JSONL 文件数据"""
    print(f"📖 读取 JSONL: {file_path}")
    
    converted_data = []
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                data = json.loads(line)
                converted_row = convert_jsonl_row(data)
                converted_data.append(converted_row)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  第 {idx + 1} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"   ⚠️  第 {idx + 1} 行转换失败: {e}")
                continue
    
    print(f"   ✅ 读取成功，共 {line_count} 行，成功转换 {len(converted_data)} 行")
    
    return converted_data


def split_dataset(data: List[Dict], train_ratio: float = 0.9, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    随机切分数据集为训练集和测试集
    
    Args:
        data: 原始数据列表
        train_ratio: 训练集比例，默认0.9
        random_seed: 随机种子
        
    Returns:
        训练集和测试集的元组 (train_data, test_data)
    """
    random.seed(random_seed)
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    split_idx = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data


def save_to_parquet(data: List[Dict], output_path: str, base_name: str, dataset_type: str, save_excel: bool = False) -> Tuple[str, Optional[str]]:
    """
    保存数据为 Parquet 格式（可选同时保存 Excel）
    
    Args:
        data: 要保存的数据列表
        output_path: 输出目录
        base_name: 基础文件名
        dataset_type: 数据集类型（train/test）
        save_excel: 是否同时保存 Excel 文件
        
    Returns:
        (Parquet 文件路径, Excel 文件路径或None)
    """
    os.makedirs(output_path, exist_ok=True)
    
    parquet_filename = f"{base_name}_{dataset_type}.parquet"
    parquet_path = os.path.join(output_path, parquet_filename)
    
    df = pd.DataFrame(data)
    df['id'] = [f"{base_name}_{dataset_type}_{i:06d}" for i in range(len(df))]
    df = df[['id', 'content', 'teacher_response']]
    
    df.to_parquet(parquet_path, index=False)
    
    excel_path = None
    if save_excel:
        excel_filename = f"{base_name}_{dataset_type}.xlsx"
        excel_path = os.path.join(output_path, excel_filename)
        # 为Excel创建可读格式：将content列表转为字符串
        excel_df = df.copy()
        excel_df['content'] = excel_df['content'].apply(lambda x: x[0]['content'] if x else '')
        excel_df.columns = ['id', 'prompt', 'teacher_response']
        excel_df.to_excel(excel_path, index=False)
    
    return parquet_path, excel_path


def detect_file_type(file_path: str) -> str:
    """检测文件类型"""
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.xlsx', '.xls']:
        return 'excel'
    elif ext in ['.jsonl', '.json']:
        return 'jsonl'
    else:
        raise ValueError(f"不支持的文件格式: {ext}，支持 .xlsx, .xls, .jsonl, .json")


def process_file(
    input_path: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.9,
    random_seed: int = 42,
    split_data: bool = True,
    save_excel: bool = False
) -> None:
    """
    处理单个文件
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录（默认为输入文件同目录下的 processed 文件夹）
        train_ratio: 训练集比例
        random_seed: 随机种子
        split_data: 是否切分数据集
        save_excel: 是否同时输出 Excel 文件
    """
    print("=" * 60)
    print("通用数据转换工具：Excel/JSONL -> GAD Parquet 格式")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        print(f"❌ 错误：文件不存在: {input_path}")
        return
    
    # 检测文件类型
    file_type = detect_file_type(input_path)
    print(f"\n📁 检测到文件类型: {file_type.upper()}")
    
    # 确定输出目录
    if output_dir is None:
        input_dir = os.path.dirname(input_path)
        output_dir = os.path.join(input_dir, "processed")
    
    # 获取基础文件名
    base_name = Path(input_path).stem
    
    # 加载数据
    if file_type == 'excel':
        data = load_excel_data(input_path)
    else:
        data = load_jsonl_data(input_path)
    
    if not data:
        print("❌ 错误：没有成功转换任何数据")
        return
    
    print(f"\n📊 成功转换 {len(data)} 条数据")
    print(f"   输出目录: {output_dir}")
    print(f"   输出Excel: {'是' if save_excel else '否'}")
    
    if split_data:
        # 切分数据集
        print(f"\n🔄 切分数据集（训练集比例: {train_ratio*100:.0f}%）...")
        train_data, test_data = split_dataset(data, train_ratio, random_seed)
        print(f"   ✅ 切分完成")
        print(f"   训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
        print(f"   测试集: {len(test_data)} 条 ({len(test_data)/len(data)*100:.1f}%)")
        
        # 保存训练集
        print(f"\n💾 保存训练集...")
        train_parquet, train_excel = save_to_parquet(train_data, output_dir, base_name, "train", save_excel)
        print(f"   ✅ Parquet: {train_parquet}")
        if train_excel:
            print(f"   ✅ Excel: {train_excel}")
        
        # 保存测试集
        print(f"\n💾 保存测试集...")
        test_parquet, test_excel = save_to_parquet(test_data, output_dir, base_name, "test", save_excel)
        print(f"   ✅ Parquet: {test_parquet}")
        if test_excel:
            print(f"   ✅ Excel: {test_excel}")
        
        # 验证
        verify_and_show_sample(train_parquet, "训练集")
        verify_and_show_sample(test_parquet, "测试集")
    else:
        # 不切分，直接保存全部数据
        print(f"\n💾 保存全部数据...")
        output_parquet, output_excel = save_to_parquet(data, output_dir, base_name, "all", save_excel)
        print(f"   ✅ Parquet: {output_parquet}")
        if output_excel:
            print(f"   ✅ Excel: {output_excel}")
        
        verify_and_show_sample(output_parquet, "全部数据")
    
    print("\n" + "=" * 60)
    print("✅ 数据转换完成！")
    print("=" * 60)


def verify_and_show_sample(parquet_path: str, dataset_name: str) -> None:
    """验证并显示示例数据"""
    try:
        df = pd.read_parquet(parquet_path)
        print(f"\n🔍 验证 {dataset_name}: {len(df)} 行，列: {df.columns.tolist()}")
        
        if len(df) > 0:
            first_row = df.iloc[0]
            print(f"\n📝 {dataset_name}第一行示例:")
            print(f"   ID: {first_row['id']}")
            print(f"   Content (前200字符): {str(first_row['content'])[:200]}...")
            print(f"   Teacher Response (前200字符): {str(first_row['teacher_response'])[:200]}...")
    except Exception as e:
        print(f"   ⚠️  {dataset_name}验证失败: {e}")


def main():
    """主函数 - 直接修改下方参数即可运行"""
    
    # ==================== 配置参数（直接修改这里）====================
    
    # 输入文件路径（支持 .xlsx, .xls, .jsonl, .json）
    INPUT_PATH = "/path/to/your/data.jsonl"
    
    # 输出目录（设为 None 则默认为输入文件同目录下的 processed 文件夹）
    OUTPUT_DIR = None
    
    # 训练集比例（0.0 ~ 1.0）
    TRAIN_RATIO = 0.9
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 是否切分数据集（True: 切分为训练集/测试集，False: 输出全部数据）
    SPLIT_DATA = True
    
    # 是否同时输出 Excel 文件（True: 输出 parquet + xlsx，False: 仅输出 parquet）
    SAVE_EXCEL = True
    
    # ================================================================
    
    process_file(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED,
        split_data=SPLIT_DATA,
        save_excel=SAVE_EXCEL
    )


if __name__ == "__main__":
    main()
