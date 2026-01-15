"""
GenRM Exhaustive Ranking (v9 Hybrid Sorting)

【核心升级】：
实现了 "Grouped Longest-First" 排序策略：
1. 先按 Prompt 将任务分组。
2. 组与组之间，按该组的"最大任务长度"降序排列（保证整体 Longest First）。
3. 组内按任务长度降序排列。

收益：同时获得 100% Cache 命中率 和 最优的尾部延迟控制。
"""
import pandas as pd
import asyncio
import aiohttp
import json
import re
import time
import numpy as np
from tqdm.asyncio import tqdm_asyncio
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations

# ============================================================================
# 1. 配置参数
# ============================================================================

DATA_PATH = "/home/jovyan/lyy/sa-training/zhijian/质检结果_history_summary_v3_all_6.xlsx"
BASE_URL = "http://10.72.1.12:8011/v1"
MODEL_NAME = "GenRM"
API_KEY = "sk-xxxx"

# B300 强力并发
CONCURRENCY = 512
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

PROMPT_COL = "prompt"
HISTORY_COL = None
RESPONSE_COLS = None

OUTPUT_DIR = "outputs/genrm_ranking_exhaustive"
MAX_SAMPLES = None

# ============================================================================
# 2. 核心代码
# ============================================================================
PERF_METRICS = []

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
    perf_stats: Dict = field(default_factory=dict)

def load_dataset(path: str) -> pd.DataFrame:
    print(f"加载数据集: {path}")
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except:
            df = pd.read_csv(path, encoding='gbk')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def auto_detect_response_cols(columns: List[str], exclude: List[str]) -> List[str]:
    exclude_set = set(c.lower() for c in exclude if c)
    return [c for c in columns if c.lower() not in exclude_set]

def generate_exhaustive_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for i, j in combinations(range(n), 2):
        pairs.append((i, j))
        pairs.append((j, i))
    return pairs

def parse_genrm_output(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not output: return None, None, None
    result = output.split("</think>")[-1].strip()
    try:
        match = re.search(r'\{[^}]+\}', result)
        if match:
            data = json.loads(match.group())
            s1 = data.get('helpfulness_response_1') or data.get('score_1')
            s2 = data.get('helpfulness_response_2') or data.get('score_2')
            sr = data.get('ranking') or data.get('ranking_score')
            if s1 and s2 and sr:
                return float(s1), float(s2), float(sr)
    except: pass
    patterns = [
        (r'helpfulness_response_1[:\s]+(\d+\.?\d*)', r'helpfulness_response_2[:\s]+(\d+\.?\d*)', r'ranking[:\s]+(\d+\.?\d*)'),
        (r'[Rr]esponse\s*1[:\s]+(\d+\.?\d*)', r'[Rr]esponse\s*2[:\s]+(\d+\.?\d*)', r'[Rr]anking[:\s]+(\d+\.?\d*)'),
    ]
    for p1, p2, p3 in patterns:
        m1, m2, m3 = re.search(p1, result, re.I), re.search(p2, result, re.I), re.search(p3, result, re.I)
        if m1 and m2 and m3: return float(m1.group(1)), float(m2.group(1)), float(m3.group(1))
    return None, None, None

def apply_tiebreaker(s_i: float, s_j: float, s_r: float) -> Tuple[float, float]:
    if s_i == s_j: return s_i + (3.5 - s_r), s_j + (s_r - 3.5)
    return s_i, s_j

async def call_genrm(session: aiohttp.ClientSession, row_idx: int, i: int, j: int,
                     prompt: str, resp_i: str, resp_j: str,
                     base_url: str, model: str, api_key: str,
                     timeout: int, retries: int) -> ComparisonResult:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "response_1", "content": resp_i},
        {"role": "response_2", "content": resp_j}
    ]
    # 保持原版参数，确保正确性
    payload = {
        "model": model, "messages": messages,
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096
    }
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for attempt in range(retries):
        t_start = time.perf_counter()
        ttft = 0.0
        try:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                t_headers = time.perf_counter()
                ttft = (t_headers - t_start) * 1000 
                if resp.status == 200:
                    data = await resp.json()
                    t_end = time.perf_counter()
                    latency_ms = (t_end - t_start) * 1000
                    
                    usage = data.get('usage', {})
                    p_len = usage.get('prompt_tokens', len(str(messages)) // 4)
                    c_len = usage.get('completion_tokens', 0)
                    if c_len == 0 and len(data['choices']) > 0:
                        c_len = len(data['choices'][0]['message']['content']) // 4

                    stats = {
                        "Prompt": int(p_len), "Completion": int(c_len),
                        "TTFT(ms)": ttft, "TPOT(ms)": latency_ms / c_len if c_len > 0 else 0,
                        "TPS": c_len / (latency_ms / 1000) if latency_ms > 0 else 0,
                        "处理时间(ms)": latency_ms, "status": "success"
                    }
                    PERF_METRICS.append(stats)

                    output = data['choices'][0]['message']['content']
                    s_i, s_j, s_r = parse_genrm_output(output)
                    if s_i is not None:
                        s_i_adj, s_j_adj = apply_tiebreaker(s_i, s_j, s_r)
                        return ComparisonResult(row_idx, i, j, s_i, s_j, s_r, s_i_adj, s_j_adj, parsed=True, perf_stats=stats)
                    return ComparisonResult(row_idx, i, j, error=f"解析失败: {output[:100]}", parsed=False, perf_stats=stats)
                else:
                    if attempt == retries - 1: return ComparisonResult(row_idx, i, j, error=f"HTTP {resp.status}", parsed=False)
        except Exception as e:
            if attempt == retries - 1: return ComparisonResult(row_idx, i, j, error=str(e)[:100], parsed=False)
        await asyncio.sleep(1)
    return ComparisonResult(row_idx, i, j, error="未知错误", parsed=False)

def hybrid_sort_tasks(tasks_info: List[Dict]) -> List[Tuple]:
    """
    【核心优化算法】Grouped Longest-First
    1. 按 Prompt 分组
    2. 计算每组的 Max Length
    3. 组间按 Max Length 降序
    4. 组内按 Length 降序
    """
    # 1. Grouping
    groups = {}
    for task in tasks_info:
        # prompt is at index 3 of the tuple in 'args'
        # task structure: {'args': (idx, i, j, prompt, ...), 'len': ...}
        prompt = task['args'][3]
        if prompt not in groups:
            groups[prompt] = []
        groups[prompt].append(task)
    
    print(f"聚合为 {len(groups)} 个 Prompt 组进行调度优化...")

    # 2. Weighting & Intra-Group Sort
    weighted_groups = []
    for prompt, group_tasks in groups.items():
        # 组内排序
        group_tasks.sort(key=lambda x: x['len'], reverse=True)
        # 计算组权重 (最大长度)
        max_len = group_tasks[0]['len']
        weighted_groups.append((max_len, group_tasks))
    
    # 3. Global Sort (按组权重降序)
    weighted_groups.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Flatten
    sorted_args = []
    for _, group_tasks in weighted_groups:
        for t in group_tasks:
            sorted_args.append(t['args'])
            
    return sorted_args

async def process_all(df: pd.DataFrame, response_cols: List[str], 
                      base_url: str, model: str, api_key: str,
                      concurrency: int, timeout: int, retries: int) -> List[Dict]:
    tasks_info = []
    row_data = {}
    
    print("正在构建任务队列...")
    for idx, row in df.iterrows():
        prompt = row.get(PROMPT_COL)
        if pd.isna(prompt): continue
        
        responses, col_mapping = [], []
        for col in response_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                responses.append(str(val))
                col_mapping.append(col)
        if len(responses) < 2: continue
        
        original = {c: row.get(c) for c in df.columns 
                    if c.lower() != PROMPT_COL.lower() and c not in response_cols}
        row_data[idx] = {'prompt': prompt, 'responses': responses, 'col_mapping': col_mapping, 'original': original}
        
        pairs = generate_exhaustive_pairs(len(responses))
        for i, j in pairs:
            task_len = len(str(prompt)) + len(str(responses[i])) + len(str(responses[j]))
            tasks_info.append({
                'args': (idx, i, j, prompt, responses[i], responses[j]),
                'len': task_len
            })
    
    print(f"共{len(row_data)}条数据，{len(tasks_info)}个比较任务 (Exhaustive)")
    
    # 【应用 Hybrid Sorting】
    print("应用 Grouped Longest-First 排序 (兼顾 Cache 与 吞吐)...")
    sorted_args = hybrid_sort_tasks(tasks_info)
    
    semaphore = asyncio.Semaphore(concurrency)
    async def run_task(session, args):
        async with semaphore:
            return await call_genrm(session, *args, base_url, model, api_key, timeout, retries)
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [run_task(session, args) for args in sorted_args]
        results = await tqdm_asyncio.gather(*coros, desc="推理中")
    
    return aggregate(results, row_data)

def aggregate(results: List[ComparisonResult], row_data: Dict) -> List[Dict]:
    """Exhaustive 聚合逻辑"""
    by_row = {}
    for r in results: by_row.setdefault(r.row_idx, []).append(r)
    
    final = []
    for idx, data in row_data.items():
        n = len(data['responses'])
        all_scores = [[] for _ in range(n)]
        wins = [0]*n; draws = [0]*n; losses = [0]*n
        details = []
        
        for r in by_row.get(idx, []):
            if r.parsed:
                all_scores[r.i].append(r.s_i_adj)
                all_scores[r.j].append(r.s_j_adj)
                if r.s_i_adj > r.s_j_adj: wins[r.i] += 1; losses[r.j] += 1
                elif r.s_i_adj < r.s_j_adj: losses[r.i] += 1; wins[r.j] += 1
                else: draws[r.i] += 1; draws[r.j] += 1
                details.append({'pair': (r.i, r.j), 's1': r.s_i, 's2': r.s_j, 'sr': r.s_r, 'parsed': True})
            else:
                details.append({'pair': (r.i, r.j), 'error': r.error, 'parsed': False})
        
        scores = []
        for i in range(n):
            avg = sum(all_scores[i]) / len(all_scores[i]) if all_scores[i] else None
            total = wins[i] + draws[i] + losses[i]
            scores.append({
                'idx': i, 'avg_score': avg, 'wins': wins[i], 'draws': draws[i], 'losses': losses[i],
                'win_rate': wins[i] / total if total > 0 else 0
            })
        
        valid = sorted([s for s in scores if s['avg_score'] is not None], key=lambda x: -x['avg_score'])
        for rank, s in enumerate(valid, 1): s['rank'] = rank
        for s in scores:
            if s['avg_score'] is None: s['rank'] = None
        
        row = {'index': idx, 'prompt': data['prompt'], 'n_responses': n}
        for i, (resp, col) in enumerate(zip(data['responses'], data['col_mapping'])):
            row[f'response_{i+1}'] = resp
            row[f'response_{i+1}_col'] = col
            row[f'response_{i+1}_avg_score'] = scores[i]['avg_score']
            row[f'response_{i+1}_rank'] = scores[i]['rank']
            row[f'response_{i+1}_wins'] = scores[i]['wins']
            row[f'response_{i+1}_draws'] = scores[i]['draws']
            row[f'response_{i+1}_losses'] = scores[i]['losses']
            row[f'response_{i+1}_win_rate'] = scores[i]['win_rate']
        
        if valid:
            best = valid[0]
            row['best_response_idx'] = best['idx'] + 1
            row['best_response_col'] = data['col_mapping'][best['idx']]
            row['best_response_avg_score'] = best['avg_score']
            row['best_response_win_rate'] = best['win_rate']
        else:
            row['best_response_idx'] = None
            
        row['comparison_details'] = json.dumps(details, ensure_ascii=False)
        for k, v in data['original'].items(): row[f'original_{k}'] = v
        final.append(row)
    return final

def print_summary(results: List[Dict], total_time: float):
    # 业务分布
    dist = {}
    success_cnt = 0
    for r in results:
        col = r.get('best_response_col')
        if col: dist[col] = dist.get(col, 0) + 1; success_cnt += 1
            
    print("\n" + "="*50)
    print("业务指标: 最佳 Response 分布")
    print("="*50)
    print(f"有效评分: {success_cnt}/{len(results)}")
    for col, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {col}: {cnt} ({cnt/success_cnt*100:.1f}%)")

    # 性能指标
    if not PERF_METRICS: return
    df_perf = pd.DataFrame(PERF_METRICS)
    print("\n" + "="*50)
    print(f"性能指标 (Hybrid Sorted) | Time: {total_time:.2f}s")
    print("="*50)
    print(f"QPS: {len(df_perf)/total_time:.2f}")
    if 'Completion' in df_perf.columns:
        print(f"Tokens/s: {df_perf['Completion'].sum()/total_time:.2f}")
    for m in ['TTFT(ms)', 'TPOT(ms)', '处理时间(ms)']:
        if m in df_perf.columns:
            d = df_perf[m].dropna()
            if len(d) > 0:
                print(f"{m:<15} | Avg: {d.mean():.1f} | P99: {np.percentile(d, 99):.1f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f'performance_data_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"details": PERF_METRICS}, f, ensure_ascii=False, indent=2)
    print(f"性能数据已保存: {json_path}")

async def main_async():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset(DATA_PATH)
    
    if RESPONSE_COLS: response_cols = RESPONSE_COLS
    else: response_cols = auto_detect_response_cols(df.columns.tolist(), [PROMPT_COL] + ([HISTORY_COL] if HISTORY_COL else []))
    
    if MAX_SAMPLES: df = df.head(MAX_SAMPLES)
    
    print(f"\n配置: 并发={CONCURRENCY}, 超时={REQUEST_TIMEOUT}s")
    
    start_time = time.perf_counter()
    results = await process_all(df, response_cols, BASE_URL, MODEL_NAME, API_KEY,
                                CONCURRENCY, REQUEST_TIMEOUT, MAX_RETRIES)
    total_time = time.perf_counter() - start_time
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f'genrm_exhaustive_hybrid_{timestamp}.xlsx')
    pd.DataFrame(results).to_excel(output_file, index=False)
    print(f"\n结果保存: {output_file}")
    
    print_summary(results, total_time)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()