"""
GenRM Circular Comparison 多Response排序脚本

基于官方算法：
1. Circular Comparison: N个回复形成环，比较相邻对 (r1,r2), (r2,r3), ..., (rN,r1)
2. 每个回复在左位置和右位置各被评估一次
3. Tiebreaker: 当s_i = s_j时，用ranking score调整
4. 最终得分 R_base = (左位置得分 + 右位置得分) / 2

评分说明：
- Helpfulness Score: 1-5，越高越好
- Ranking Score: 1-6 (1=Response1远好于2, 6=Response2远好于1, 3.5=中性)

输出Excel列：
- index, prompt, n_responses: 基础信息
- response_N, response_N_col, response_N_R_base, response_N_rank: 每个response的信息
- best_response_idx, best_response_col, best_response_R_base: 最佳response
- comparison_details: JSON格式的详细比较记录
- original_xxx: 原始数据的其他列
"""
import pandas as pd
import asyncio
import aiohttp
import json
import re
from tqdm.asyncio import tqdm_asyncio
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# 配置参数
# ============================================================================

# 数据集路径（支持 xlsx, csv, parquet）
DATA_PATH = "/home/jovyan/JQ/gad_gspo_B300/data/test_data/GenRM_test/82val_user_profile_20260105_193611.xlsx"
BASE_URL = "http://10.72.1.12:8011/v1"
MODEL_NAME = "GenRM"
API_KEY = "sk-xxxx"               # API密钥

CONCURRENCY = 20                # 并发数
MAX_RETRIES = 3                 # 重试次数
REQUEST_TIMEOUT = 300           # 超时(秒)

PROMPT_COL = "prompt"           # prompt列名
HISTORY_COL = None              # 历史对话列名(可选)
RESPONSE_COLS = None            # response列名列表，None则自动检测

OUTPUT_DIR = "outputs/genrm_ranking"
MAX_SAMPLES = None              # 限制样本数(测试用)

# ============================================================================
# 核心代码
# ============================================================================

@dataclass
class ComparisonResult:
    row_idx: int
    i: int
    j: int
    s_i: Optional[float] = None
    s_j: Optional[float] = None
    s_r: Optional[float] = None
    s_i_adj: Optional[float] = None
    s_j_adj: Optional[float] = None
    error: Optional[str] = None
    parsed: bool = False


def load_dataset(path: str) -> pd.DataFrame:
    print(f"加载数据集: {path}")
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print(f"列名: {df.columns.tolist()}, 共{len(df)}条")
    return df


def auto_detect_response_cols(columns: List[str], exclude: List[str]) -> List[str]:
    """自动检测response列：排除prompt和history列后的所有列"""
    exclude_set = set(c.lower() for c in exclude if c)
    return [c for c in columns if c.lower() not in exclude_set]


def parse_genrm_output(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """解析GenRM输出，返回(s1, s2, ranking)"""
    if not output:
        return None, None, None
    
    result = output.split("</think>")[-1].strip()
    
    # JSON解析
    try:
        match = re.search(r'\{[^}]+\}', result)
        if match:
            data = json.loads(match.group())
            s1 = data.get('helpfulness_response_1') or data.get('score_1')
            s2 = data.get('helpfulness_response_2') or data.get('score_2')
            sr = data.get('ranking') or data.get('ranking_score')
            if s1 and s2 and sr:
                return float(s1), float(s2), float(sr)
    except:
        pass
    
    # 正则匹配
    patterns = [
        (r'helpfulness_response_1[:\s]+(\d+\.?\d*)', r'helpfulness_response_2[:\s]+(\d+\.?\d*)', r'ranking[:\s]+(\d+\.?\d*)'),
        (r'[Rr]esponse\s*1[:\s]+(\d+\.?\d*)', r'[Rr]esponse\s*2[:\s]+(\d+\.?\d*)', r'[Rr]anking[:\s]+(\d+\.?\d*)'),
    ]
    for p1, p2, p3 in patterns:
        m1, m2, m3 = re.search(p1, result, re.I), re.search(p2, result, re.I), re.search(p3, result, re.I)
        if m1 and m2 and m3:
            return float(m1.group(1)), float(m2.group(1)), float(m3.group(1))
    
    return None, None, None


def apply_tiebreaker(s_i: float, s_j: float, s_r: float) -> Tuple[float, float]:
    """Tiebreaker: 当分数相等时用ranking调整"""
    if s_i == s_j:
        return s_i + (3.5 - s_r), s_j + (s_r - 3.5)
    return s_i, s_j


async def call_genrm(session: aiohttp.ClientSession, row_idx: int, i: int, j: int,
                     prompt: str, resp_i: str, resp_j: str,
                     base_url: str, model: str, api_key: str,
                     timeout: int, retries: int) -> ComparisonResult:
    """调用GenRM比较两个response"""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "response_1", "content": resp_i},
        {"role": "response_2", "content": resp_j}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384
    }
    
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output = data['choices'][0]['message']['content']
                    s_i, s_j, s_r = parse_genrm_output(output)
                    
                    if s_i is not None and s_j is not None and s_r is not None:
                        s_i_adj, s_j_adj = apply_tiebreaker(s_i, s_j, s_r)
                        return ComparisonResult(row_idx, i, j, s_i, s_j, s_r, s_i_adj, s_j_adj, parsed=True)
                    return ComparisonResult(row_idx, i, j, error=f"解析失败: {output[:100]}", parsed=False)
                else:
                    err = await resp.text()
                    if attempt == retries - 1:
                        return ComparisonResult(row_idx, i, j, error=f"HTTP {resp.status}: {err[:100]}", parsed=False)
        except asyncio.TimeoutError:
            if attempt == retries - 1:
                return ComparisonResult(row_idx, i, j, error="超时", parsed=False)
        except Exception as e:
            if attempt == retries - 1:
                return ComparisonResult(row_idx, i, j, error=str(e)[:100], parsed=False)
        await asyncio.sleep(1)
    
    return ComparisonResult(row_idx, i, j, error="未知错误", parsed=False)


async def process_all(df: pd.DataFrame, response_cols: List[str], 
                      base_url: str, model: str, api_key: str,
                      concurrency: int, timeout: int, retries: int) -> List[Dict]:
    """处理所有数据"""
    # 准备任务
    tasks_info = []
    row_data = {}
    
    for idx, row in df.iterrows():
        prompt = row.get(PROMPT_COL)
        if pd.isna(prompt):
            continue
        
        responses, col_mapping = [], []
        for col in response_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                responses.append(str(val))
                col_mapping.append(col)
        
        if len(responses) < 2:
            continue
        
        original = {c: row.get(c) for c in df.columns 
                    if c not in [PROMPT_COL] + response_cols + ([HISTORY_COL] if HISTORY_COL else [])}
        row_data[idx] = {'prompt': prompt, 'responses': responses, 'col_mapping': col_mapping, 'original': original}
        
        n = len(responses)
        for i in range(n):
            j = (i + 1) % n
            tasks_info.append((idx, i, j, prompt, responses[i], responses[j]))
    
    print(f"共{len(row_data)}条数据，{len(tasks_info)}个比较任务")
    
    # 并发执行
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def run_task(session, info):
        async with semaphore:
            return await call_genrm(session, *info, base_url, model, api_key, timeout, retries)
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [run_task(session, info) for info in tasks_info]
        results = await tqdm_asyncio.gather(*coros, desc="处理中")
    
    return aggregate(results, row_data)


def aggregate(results: List[ComparisonResult], row_data: Dict) -> List[Dict]:
    """聚合比较结果，计算排名"""
    # 按行分组
    by_row = {}
    for r in results:
        by_row.setdefault(r.row_idx, []).append(r)
    
    final = []
    for idx, data in row_data.items():
        n = len(data['responses'])
        scores_left = [None] * n
        scores_right = [None] * n
        details = []
        
        for r in by_row.get(idx, []):
            if r.parsed:
                scores_left[r.i] = r.s_i_adj
                scores_right[r.j] = r.s_j_adj
                details.append({'pair': (r.i, r.j), 's1': r.s_i, 's2': r.s_j, 'sr': r.s_r, 'parsed': True})
            else:
                details.append({'pair': (r.i, r.j), 'error': r.error, 'parsed': False})
        
        # 计算R_base
        scores = []
        for i in range(n):
            left, right = scores_left[i], scores_right[i]
            if left is not None and right is not None:
                avg = (left + right) / 2
            elif left is not None:
                avg = left
            elif right is not None:
                avg = right
            else:
                avg = None
            scores.append({'idx': i, 'R_base': avg, 'left': left, 'right': right})
        
        # 排名
        valid = sorted([s for s in scores if s['R_base'] is not None], key=lambda x: -x['R_base'])
        for rank, s in enumerate(valid, 1):
            s['rank'] = rank
        for s in scores:
            if s['R_base'] is None:
                s['rank'] = None
        
        # 构建输出
        row = {'index': idx, 'prompt': data['prompt'], 'n_responses': n}
        for i, (resp, col) in enumerate(zip(data['responses'], data['col_mapping'])):
            row[f'response_{i+1}'] = resp
            row[f'response_{i+1}_col'] = col
            row[f'response_{i+1}_R_base'] = scores[i]['R_base']
            row[f'response_{i+1}_rank'] = scores[i]['rank']
        
        if valid:
            best = valid[0]
            row['best_response_idx'] = best['idx'] + 1
            row['best_response_col'] = data['col_mapping'][best['idx']]
            row['best_response_R_base'] = best['R_base']
        
        row['comparison_details'] = json.dumps(details, ensure_ascii=False)
        for k, v in data['original'].items():
            row[f'original_{k}'] = v
        
        final.append(row)
    
    return final


async def main_async():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset(DATA_PATH)
    
    # 检测response列
    response_cols = RESPONSE_COLS
    if not response_cols:
        exclude = [PROMPT_COL] + ([HISTORY_COL] if HISTORY_COL else [])
        response_cols = auto_detect_response_cols(df.columns.tolist(), exclude)
        print(f"自动检测到{len(response_cols)}个Response列: {response_cols}")
    
    if len(response_cols) < 2:
        print(f"错误: response列不足2个")
        return
    
    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)
    
    print(f"\n配置: 并发={CONCURRENCY}, 超时={REQUEST_TIMEOUT}s")
    
    results = await process_all(df, response_cols, BASE_URL, MODEL_NAME, API_KEY,
                                CONCURRENCY, REQUEST_TIMEOUT, MAX_RETRIES)
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f'genrm_ranking_{timestamp}.xlsx')
    pd.DataFrame(results).to_excel(output_file, index=False)
    print(f"\n完成！保存到: {output_file}")
    
    # 统计
    success = sum(1 for r in results if r.get('best_response_idx'))
    print(f"成功: {success}/{len(results)}")
    
    # 最佳分布
    dist = {}
    for r in results:
        col = r.get('best_response_col')
        if col:
            dist[col] = dist.get(col, 0) + 1
    if dist:
        print("最佳Response分布:")
        for col, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {col}: {cnt} ({cnt/success*100:.1f}%)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
