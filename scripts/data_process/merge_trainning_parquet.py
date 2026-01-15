#!/usr/bin/env python3
"""
合并多个训练集 parquet 文件的脚本
支持任意数量的 parquet 文件合并，可选择随机打乱和切分训练/测试集

直接修改下方配置参数后运行：
python scripts/data_process/merge_trainning_parquet.py
"""

import pandas as pd
import os
from typing import List, Tuple
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """读取 parquet 文件"""
    try:
        logger.info(f"正在读取文件: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"成功读取文件，包含 {len(df)} 行数据")
        return df
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        raise


def validate_data_structure(df: pd.DataFrame, file_name: str) -> bool:
    """验证数据结构是否符合预期"""
    expected_columns = ["content", "teacher_response"]

    if not all(col in df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df.columns]
        logger.warning(f"{file_name} 缺少列: {missing_cols}")
        logger.info(f"{file_name} 实际列名: {list(df.columns)}")
        return False

    logger.info(f"{file_name} 数据结构验证通过，形状: {df.shape}")
    return True


def add_dataset_identifier(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """为数据集添加标识符"""
    df_copy = df.copy()
    df_copy["dataset_source"] = dataset_name
    df_copy["id"] = [f"{dataset_name}_{i:06d}" for i in range(len(df_copy))]
    return df_copy


def merge_datasets(dfs: List[pd.DataFrame], dataset_names: List[str]) -> pd.DataFrame:
    """合并多个数据集"""
    logger.info("开始合并数据集...")

    processed_dfs = []
    for df, name in zip(dfs, dataset_names):
        processed_df = add_dataset_identifier(df, name)
        processed_dfs.append(processed_df)
        logger.info(f"  {name}: {len(df)} 行数据")

    merged_df = pd.concat(processed_dfs, ignore_index=True)
    logger.info(f"合并完成，总共 {len(merged_df)} 行数据")

    return merged_df


def shuffle_dataset(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """打乱数据集顺序"""
    logger.info("正在打乱数据集顺序...")
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    logger.info("数据集打乱完成")
    return shuffled_df


def sample_test_set(
    df: pd.DataFrame, test_ratio: float = 0.02, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从合并后的数据集中采样测试集"""
    logger.info(f"正在从数据集中采样 {test_ratio*100}% 作为测试集...")

    test_size = max(1, int(len(df) * test_ratio))
    logger.info(f"测试集大小: {test_size} 行")

    test_df = df.sample(n=test_size, random_state=random_seed)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"采样完成 - 训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")

    if "dataset_source" in test_df.columns:
        logger.info("测试集数据来源分布:")
        test_source_counts = test_df["dataset_source"].value_counts()
        for source, count in test_source_counts.items():
            percentage = (count / len(test_df)) * 100
            logger.info(f"  {source}: {count} 行 ({percentage:.1f}%)")

    return train_df, test_df


def save_merged_dataset(df: pd.DataFrame, output_path: str) -> None:
    """保存合并后的数据集"""
    try:
        logger.info(f"正在保存合并后的数据集到: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"成功保存合并后的数据集，包含 {len(df)} 行数据")
        if "dataset_source" in df.columns:
            logger.info("数据集统计信息:")
            source_counts = df["dataset_source"].value_counts()
            for source, count in source_counts.items():
                logger.info(f"  {source}: {count} 行")

    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise


def preview_samples(df: pd.DataFrame, num_samples: int = 3, random_seed: int = 42) -> None:
    """抽样预览合并后的数据"""
    print("\n" + "=" * 80)
    print(f"📋 抽样预览（随机抽取 {num_samples} 条数据）")
    print("=" * 80)

    samples = df.sample(n=min(num_samples, len(df)), random_state=random_seed)

    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"【样本 {idx}】")
        print(f"  ID: {row.get('id', 'N/A')}")
        print(f"  来源: {row.get('dataset_source', 'N/A')}")

        # 处理 content 字段
        content = row.get("content", "")
        if isinstance(content, list) and len(content) > 0:
            prompt = content[0].get("content", "") if isinstance(content[0], dict) else str(content[0])
        else:
            prompt = str(content)

        # 截断显示
        max_len = 300
        prompt_display = prompt[:max_len] + "..." if len(prompt) > max_len else prompt
        response = str(row.get("teacher_response", ""))
        response_display = response[:max_len] + "..." if len(response) > max_len else response

        print(f"  Prompt ({len(prompt)} 字符):")
        print(f"    {prompt_display}")
        print(f"  Teacher Response ({len(response)} 字符):")
        print(f"    {response_display}")

    print("\n" + "=" * 80)


def save_split_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str, base_name: str
) -> None:
    """保存切分后的训练集和测试集"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, f"{base_name}_train.parquet")
        train_df.to_parquet(train_path, index=False)
        logger.info(f"训练集已保存到: {train_path} ({len(train_df)} 行)")

        test_path = os.path.join(output_dir, f"{base_name}_test.parquet")
        test_df.to_parquet(test_path, index=False)
        logger.info(f"测试集已保存到: {test_path} ({len(test_df)} 行)")

        logger.info("数据集切分完成统计:")
        total = len(train_df) + len(test_df)
        logger.info(f"总数据量: {total} 行")
        logger.info(f"训练集比例: {len(train_df) / total * 100:.1f}%")
        logger.info(f"测试集比例: {len(test_df) / total * 100:.1f}%")

        if "dataset_source" in train_df.columns:
            logger.info("\n训练集数据来源:")
            for source, count in train_df["dataset_source"].value_counts().items():
                logger.info(f"  {source}: {count} 行")

            logger.info("\n测试集数据来源:")
            for source, count in test_df["dataset_source"].value_counts().items():
                logger.info(f"  {source}: {count} 行")

    except Exception as e:
        logger.error(f"保存切分数据集失败: {e}")
        raise


def main():
    """主函数 - 直接修改下方参数即可运行"""

    # ==================== 配置参数（直接修改这里）====================

    # 输入文件列表：(文件路径, 数据集名称)
    # 可以添加任意数量的 parquet 文件
    INPUT_FILES = [
        (
            "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/reply_generation/processed/reply_id_0a1_train.parquet",
            "dataset1",
        ),
        (
            "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/reply_generation/processed/reply_id_1_1_train.parquet",
            "dataset2",
        ),
        (
            "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/reply_generation/processed/reply_id_9_train.parquet",
            "dataset3",
        ),
        (
            "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/semantic_understanding/processed/semantic_understanding-id-8_2_train.parquet",
            "dataset4",
        ),
        (
            "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/semantic_understanding/processed/semantic_understanding_1224_train.parquet",
            "dataset5",
        ),
        # 添加更多文件只需继续添加元组即可
        # ("/path/to/dataset3_train.parquet", "dataset3"),
    ]

    # 输出路径（如果切分，会自动生成 _train.parquet 和 _test.parquet）
    OUTPUT_PATH = "/home/jovyan/JQ/gad_gspo_B300/scripts/data_process/trainning_data/merged/merge-1225.parquet"

    # 是否打乱数据集顺序
    SHUFFLE = True

    # 随机种子
    RANDOM_SEED = 42

    # 是否切分训练集和测试集
    SPLIT_TRAIN_TEST = True

    # 测试集比例（仅在 SPLIT_TRAIN_TEST=True 时生效）
    TEST_RATIO = 0.02

    # 是否预览抽样数据
    PREVIEW_SAMPLES = True

    # 预览抽样数量
    PREVIEW_NUM = 3

    # ================================================================

    try:
        # 检查输入文件是否存在
        for file_path, name in INPUT_FILES:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return

        # 读取所有数据集
        dfs = []
        names = []
        for file_path, name in INPUT_FILES:
            df = read_parquet_file(file_path)
            validate_data_structure(df, name)
            dfs.append(df)
            names.append(name)

        # 合并数据集
        merged_df = merge_datasets(dfs, names)

        # 可选：打乱数据集
        if SHUFFLE:
            merged_df = shuffle_dataset(merged_df, RANDOM_SEED)

        # 抽样预览
        if PREVIEW_SAMPLES:
            preview_samples(merged_df, PREVIEW_NUM, RANDOM_SEED)

        # 根据参数决定是否切分训练集和测试集
        if SPLIT_TRAIN_TEST:
            logger.info(f"正在按 {TEST_RATIO*100}% 比例切分训练集和测试集...")
            train_df, test_df = sample_test_set(merged_df, TEST_RATIO, RANDOM_SEED)

            output_dir = os.path.dirname(OUTPUT_PATH)
            base_name = os.path.splitext(os.path.basename(OUTPUT_PATH))[0]
            save_split_datasets(train_df, test_df, output_dir, base_name)

            logger.info("数据集合并和切分完成！")
            logger.info(f"训练集: {os.path.join(output_dir, f'{base_name}_train.parquet')}")
            logger.info(f"测试集: {os.path.join(output_dir, f'{base_name}_test.parquet')}")
        else:
            save_merged_dataset(merged_df, OUTPUT_PATH)
            logger.info("数据集合并完成！")
            logger.info(f"输出文件: {OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"合并过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
