#!/usr/bin/env python3
"""检查 Student 和 Teacher response 的 token 差异"""

from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

# Student 和 Teacher 的前几个 tokens
student_tokens = [515, 262, 330]
teacher_tokens = [90, 257, 330]

print("=" * 80)
print("Token 解码对比")
print("=" * 80)

print("\nStudent 前 3 个 tokens:")
for i, token_id in enumerate(student_tokens):
    decoded = tokenizer.decode([token_id])
    print(f"  Token {i}: {token_id:6d} -> {repr(decoded)}")

print("\nTeacher 前 3 个 tokens:")
for i, token_id in enumerate(teacher_tokens):
    decoded = tokenizer.decode([token_id])
    print(f"  Token {i}: {token_id:6d} -> {repr(decoded)}")

print("\n" + "=" * 80)
print("完整解码对比")
print("=" * 80)

# 完整的 token 序列
student_full = [515, 262, 330, 101042, 100178, 788, 330, 105051, 15946, 100692, 104496, 2073, 35946, 110052, 105303, 33590, 62926, 53481, 34187, 105303, 102220, 99666, 101224, 3837, 29524, 2073, 30709, 99808, 99679, 34187, 3837, 99232, 74763, 101467, 3837, 109125, 99801, 105098, 99880, 33590, 43288, 102406, 104283, 28946, 102068, 112795, 105303, 1773, 101886, 3837, 109487, 32, 20412, 105045, 63276, 262, 330, 102349, 788, 330, 32, 13, 71928, 96, 33956, 698, 92, 151645]

teacher_full = [90, 257, 330, 101042, 100178, 788, 330, 105051, 15946, 104283, 28946, 100692, 104496, 2073, 99528, 5373, 100039, 5373, 100057, 5373, 99949, 100802, 105419, 3837, 35946, 110052, 105303, 33590, 43288, 57218, 86119, 101047, 113353, 2073, 14009, 35946, 527, 112795, 105303, 854, 100313, 100372, 101266, 1773, 497, 257, 330, 102349, 788, 330, 32, 13, 71928, 96, 33956, 1, 335]

print("\nStudent 完整文本:")
student_text = tokenizer.decode(student_full, skip_special_tokens=False)
print(repr(student_text))

print("\nTeacher 完整文本:")
teacher_text = tokenizer.decode(teacher_full, skip_special_tokens=False)
print(repr(teacher_text))

print("\n" + "=" * 80)
print("差异分析")
print("=" * 80)

# 检查是否包含换行符
student_has_newline = '\n' in student_text
teacher_has_newline = '\n' in teacher_text

print(f"\nStudent 包含换行符: {student_has_newline}")
print(f"Teacher 包含换行符: {teacher_has_newline}")

if student_has_newline:
    print(f"\nStudent 换行符位置:")
    for i, char in enumerate(student_text):
        if char == '\n':
            print(f"  位置 {i}: 前后文本 = {repr(student_text[max(0,i-5):i+6])}")

if teacher_has_newline:
    print(f"\nTeacher 换行符位置:")
    for i, char in enumerate(teacher_text):
        if char == '\n':
            print(f"  位置 {i}: 前后文本 = {repr(teacher_text[max(0,i-5):i+6])}")

print("\n" + "=" * 80)
