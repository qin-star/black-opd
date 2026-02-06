# GAD 格式奖励优化说明

## 优化日期
2026-01-29

## 优化内容

### 1. 引号风格检测功能 ✅

#### 新增功能
- **自适应引号检测**：根据 Ground Truth 的引号风格来判断 solution 是否符合要求
- **支持单引号JSON**：`{'answer': 'A'}` 和 `{"answer": "A"}` 都能正确识别
- **混合引号检测**：`{"answer": '是'}` 会被识别为混合风格并惩罚

#### 惩罚机制
```python
# 引号风格不匹配
if gt_style != sol_style:
    penalty = 0.15
    type = "quote_style_mismatch"

# 混合引号（GT 不是混合）
if sol_style == "mixed" and gt_style != "mixed":
    penalty = 0.2
    type = "quote_style_mixed"

# 混合引号（GT 也是混合）
if sol_style == "mixed" and gt_style == "mixed":
    penalty = 0.1  # 惩罚较轻
    type = "quote_style_mixed"
```

### 2. JSON 提取优化 ✅

#### 问题修复
**原问题**：`extract_json_from_text` 只检测双引号来判断字符串状态
```python
# 旧代码
if char == '"' and not escape_next:  # 只检测双引号
    in_string = not in_string
```

**优化后**：支持单引号和双引号
```python
# 新代码
if char in ('"', "'") and not escape_next:
    if not in_string:
        in_string = True
        string_char = char  # 记录引号类型
    elif char == string_char:
        in_string = False
        string_char = None
```

### 3. 引号转换优化 ✅

#### 问题修复
**原问题**：简单的 `replace("'", '"')` 会误转换字符串值中的单引号
```python
# 错误示例
{'name': "it's ok"}  # 值中的单引号不应该被转换
# 简单替换后变成：{"name": "it"s ok"}  # 语法错误！
```

**优化后**：智能转换，只转换JSON结构的引号
```python
def _convert_single_to_double_quotes(json_str: str) -> str:
    """智能转换单引号为双引号（避免转换字符串值中的引号）"""
    result = []
    in_string = False
    string_char = None
    
    for char in json_str:
        if char in ('"', "'"):
            if not in_string:
                in_string = True
                string_char = char
                result.append('"')  # 统一转换为双引号
            elif char == string_char:
                in_string = False
                result.append('"')
            else:
                result.append(char)  # 字符串内部的引号保持原样
        else:
            result.append(char)
    
    return ''.join(result)
```

### 4. 引号风格统计优化 ✅

#### 问题修复
**原问题**：统计所有引号，包括字符串值内的引号
```python
# 错误示例
{"name": "it's ok"}  # 被识别为 mixed（错误！）
# 因为统计了值中的单引号
```

**优化后**：只统计JSON结构的引号
```python
def detect_quote_style(text: str) -> Optional[str]:
    """只统计JSON结构的引号，不统计字符串值内的引号"""
    for char in json_str:
        if char in ('"', "'"):
            if not in_string:
                # 进入字符串，统计这个引号
                if char == "'":
                    single_count += 1
                else:
                    double_count += 1
            elif char == string_char:
                # 退出字符串，统计这个引号
                # 字符串内部的引号不统计
```

### 5. 混合引号惩罚逻辑优化 ✅

#### 优化前
```python
# 问题：混合引号可能被惩罚两次
if gt_style != sol_style:
    penalty = 0.15
elif sol_style == "mixed":
    penalty = 0.2
```

#### 优化后
```python
# 优先级：混合引号 > 风格不匹配
if sol_style == "mixed":
    if gt_style == "mixed":
        penalty = 0.1  # GT 也是混合的，惩罚较轻
    else:
        penalty = 0.2  # GT 不是混合的，惩罚较重
elif gt_style != sol_style:
    penalty = 0.15  # 风格不匹配
```

## 测试验证

### 测试用例
```python
# 基本匹配
GT: {'enough': '否'}
Sol: {'enough': '是'}  ✓ 无惩罚（单引号匹配）
Sol: {"enough": "是"}  ✗ 惩罚 0.15（风格不匹配）

# 混合引号
Sol: {"enough": '是'}  ✗ 惩罚 0.2（混合引号）
Sol: {'enough': "是"}  ✗ 惩罚 0.2（混合引号）

# 字符串值中包含引号
{"name": "it's ok"}    ✓ 识别为 double（正确！）
{'name': 'it\'s ok'}   ✓ 识别为 single（正确！）
```

### 测试结果
```
✓ 标准双引号JSON
✓ 标准单引号JSON
✓ 混合引号（外双内单）
✓ 混合引号（外单内双）
✓ 双引号JSON，值中有单引号
✓ 单引号JSON，值中有转义单引号
✓ 双引号JSON，值中有转义双引号
✓ 多字段双引号
✓ 多字段单引号

测试结果: 9 通过, 0 失败
```

## 优化效果

### 1. 更准确的格式检测
- 不再误判字符串值中的引号
- 正确识别混合引号情况
- 支持单引号JSON格式

### 2. 更合理的惩罚机制
- 根据GT自适应调整惩罚
- 避免重复惩罚
- 惩罚力度分级（0.1 / 0.15 / 0.2）

### 3. 更好的训练效果
- 模型学会保持与GT一致的引号风格
- 鼓励输出风格统一的JSON
- 减少格式错误

## 文件位置
- 主文件：`verl/verl/utils/reward_score/gad_format_reward.py`
- 测试文件：`tools/test_optimized_quote_detection.py`

## 版本信息
- 版本：v4 (引号风格检测版)
- 兼容性：向后兼容，不影响现有功能
