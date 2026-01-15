#!/usr/bin/env python3
"""
测试 TensorBoard 日志记录是否正常工作
"""

import os
import time
from torch.utils.tensorboard import SummaryWriter

def test_tensorboard_logging():
    # 设置 TensorBoard 目录
    backend = "fsdp2"
    experiment_name = f"black-opd-A3b-gspo-warmup-{backend}"
    tensorboard_dir = f"/home/jovyan/JQ/gad_gspo_B300/tensorboard/{experiment_name}"
    
    print(f"TensorBoard 目录: {tensorboard_dir}")
    
    # 确保目录存在
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 检查目录权限
    if not os.access(tensorboard_dir, os.W_OK):
        print(f"❌ 目录没有写权限: {tensorboard_dir}")
        return False
    
    print(f"✅ 目录权限正常")
    
    # 创建 SummaryWriter
    try:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"✅ SummaryWriter 创建成功")
        
        # 写入一些测试数据
        for i in range(10):
            writer.add_scalar('test/loss', 1.0 / (i + 1), i)
            writer.add_scalar('test/accuracy', i * 0.1, i)
            time.sleep(0.1)
        
        writer.close()
        print(f"✅ 测试数据写入完成")
        
        # 检查文件是否生成
        files = os.listdir(tensorboard_dir)
        print(f"生成的文件: {files}")
        
        if any(f.startswith('events.out.tfevents') for f in files):
            print(f"✅ TensorBoard 事件文件生成成功")
            return True
        else:
            print(f"❌ 没有生成 TensorBoard 事件文件")
            return False
            
    except Exception as e:
        print(f"❌ SummaryWriter 创建失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试 TensorBoard 日志记录...")
    success = test_tensorboard_logging()
    
    if success:
        print("\n🎉 TensorBoard 日志记录测试成功！")
        print("可以启动 TensorBoard 查看:")
        print("tensorboard --logdir=/home/jovyan/JQ/gad_gspo_B300/tensorboard/black-opd-A3b-gspo-warmup-fsdp2 --port=6019")
    else:
        print("\n❌ TensorBoard 日志记录测试失败")