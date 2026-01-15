"""
GenRM 穷举双向比较排序脚本 (Exhaustive Bidirectional Pairwise Comparison)

特点：
1. 穷举所有配对：N个response产生 N*(N-1)/2 个配对
2. 双向比较：每对进行2次比较，交换位置，消除GenRM的位置偏好
3. 最终比较次数：N*(N-1) 次

比较次数对比：
- 5个response: 环形5次 vs 穷举单向10次 vs 穷举双向20次
- 10个response: 环形10次 vs 穷举单向45次 vs 穷举双向90次

位置偏好消除原理：
- 比较(A,B): A在response_1位置，B在response_2位置 → 得到 score_A1, score_B2
- 比较(B,A): B在response_1位置，A在response_2位置 → 得到 score_B1, score_A2
- A的最终得分 = (score_A1 + score_A2) / 2
- 这样无论模型偏好哪个位置，都会被平均掉

输出Excel列：
- response_N_avg_score: 平均得分（消除位置偏好后）
- response_N_wins/draws/losses: 胜/平/负场次
- response_N_win_rate: 胜率
- response_N_rank: 最终排名
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
from itertools import combinations

# ============================================================================
# 配置参数
# ============================================================================

DATA_PATH = "/home/jovyan/JQ/gad_gspo_B300/data/test_data/GenRM_test/1-12/质检结果_rewritten_query_v3_all_6.xlsx"
BASE_URL = "http://10.72.1.12:8011/v1"
MODEL_NAME = "GenRM"
API_KEY = "sk-xxxx"

CONCURRENCY = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

PROMPT_COL = "prompt"
HISTORY_COL = None
RESPONSE_COLS = None            # None则自动检测

OUTPUT_DIR = "/home/jovyan/JQ/gad_gspo_B300/data/test_data/GenRM_test/genrm_ranking"
MAX_SAMPLES = None

# ============================================================================
# 核心代码
# ============================================================================

@dataclass
class ComparisonResult:
    row_idx: int
    i: int  # 左位置response索引
    j: int  # 右位置response索引
    s_i: Optional[float] = None  # response_i的原始分数
    s_j: Optional[float] = None  # response_j的原始分数
    s_r: Optional[float] = None  # ranking score
    s_i_adj: Optional[float] = None  # 调整后分数
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
    exclude_set = set(c.lower() for c in exclude if c)
    return [c for c in columns if c.lower() not in exclude_set]


def generate_exhaustive_pairs(n: int) -> List[Tuple[int, int]]:
    """
    生成所有可能的配对（双向比较，消除位置偏好）
    
    对于每对(i,j)，生成两个比较：
    - (i, j): i在response_1位置，j在response_2位置
    - (j, i): j在response_1位置，i在response_2位置
    
    这样可以消除GenRM模型可能存在的位置偏好
    """
    pairs = []
    for i, j in combinations(range(n), 2):
        pairs.append((i, j))  # i在左，j在右
        pairs.append((j, i))  # j在左，i在右
    return pairs


def parse_genrm_output(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not output:
        return None, None, None
    
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
    except:
        pass
    
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
    if s_i == s_j:
        return s_i + (3.5 - s_r), s_j + (s_r - 3.5)
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
    """处理所有数据（穷举比较）"""
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
        
        # 生成穷举比较对
        pairs = generate_exhaustive_pairs(len(responses))
        for i, j in pairs:
            tasks_info.append((idx, i, j, prompt, responses[i], responses[j]))
    
    n_responses = len(row_data[list(row_data.keys())[0]]['responses']) if row_data else 0
    n_pairs_per_row = len(generate_exhaustive_pairs(n_responses))
    print(f"共{len(row_data)}条数据，每条{n_responses}个response")
    print(f"双向比较模式: 每对比较2次（消除位置偏好），共{n_pairs_per_row}个比较任务/条")
    print(f"总计{len(tasks_info)}个比较任务")
    
    # 并发执行
    semaphore = asyncio.Semaphore(concurrency)
    
    async def run_task(session, info):
        async with semaphore:
            return await call_genrm(session, *info, base_url, model, api_key, timeout, retries)
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [run_task(session, info) for info in tasks_info]
        results = await tqdm_asyncio.gather(*coros, desc="穷举比较中")
    
    return aggregate(results, row_data)


def aggregate(results: List[ComparisonResult], row_data: Dict) -> List[Dict]:
    """聚合穷举比较结果"""
    by_row = {}
    for r in results:
        by_row.setdefault(r.row_idx, []).append(r)
    
    final = []
    for idx, data in row_data.items():
        n = len(data['responses'])
        
        # 每个response的所有得分
        all_scores = [[] for _ in range(n)]
        # 胜/平/负统计
        wins = [0] * n
        draws = [0] * n
        losses = [0] * n
        details = []
        
        for r in by_row.get(idx, []):
            if r.parsed:
                # 记录得分
                all_scores[r.i].append(r.s_i_adj)
                all_scores[r.j].append(r.s_j_adj)
                
                # 统计胜负
                if r.s_i_adj > r.s_j_adj:
                    wins[r.i] += 1
                    losses[r.j] += 1
                elif r.s_i_adj < r.s_j_adj:
                    losses[r.i] += 1
                    wins[r.j] += 1
                else:
                    draws[r.i] += 1
                    draws[r.j] += 1
                
                details.append({
                    'pair': (r.i, r.j), 
                    's1': r.s_i, 's2': r.s_j, 'sr': r.s_r,
                    's1_adj': r.s_i_adj, 's2_adj': r.s_j_adj,
                    'parsed': True
                })
            else:
                details.append({'pair': (r.i, r.j), 'error': r.error, 'parsed': False})
        
        # 计算平均得分
        scores = []
        for i in range(n):
            if all_scores[i]:
                avg = sum(all_scores[i]) / len(all_scores[i])
            else:
                avg = None
            scores.append({
                'idx': i, 
                'avg_score': avg,
                'n_comparisons': len(all_scores[i]),
                'wins': wins[i],
                'draws': draws[i],
                'losses': losses[i],
                'win_rate': wins[i] / (wins[i] + draws[i] + losses[i]) if (wins[i] + draws[i] + losses[i]) > 0 else 0
            })
        
        # 按平均得分排名
        valid = sorted([s for s in scores if s['avg_score'] is not None], key=lambda x: -x['avg_score'])
        for rank, s in enumerate(valid, 1):
            s['rank'] = rank
        for s in scores:
            if s['avg_score'] is None:
                s['rank'] = None
        
        # 构建输出
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
        
        row['comparison_details'] = json.dumps(details, ensure_ascii=False)
        for k, v in data['original'].items():
            row[f'original_{k}'] = v
        
        final.append(row)
    
    return final


async def main_async():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset(DATA_PATH)
    
    response_cols = RESPONSE_COLS
    if not response_cols:
        exclude = [PROMPT_COL] + ([HISTORY_COL] if HISTORY_COL else [])
        response_cols = auto_detect_response_cols(df.columns.tolist(), exclude)
        print(f"自动检测到{len(response_cols)}个Response列: {response_cols}")
    
    if len(response_cols) < 2:
        print("错误: response列不足2个")
        return
    
    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)
    
    # 计算比较次数
    n = len(response_cols)
    pairs_per_row = n * (n - 1)  # 双向比较
    print(f"\n穷举双向模式: {n}个response，每对比较2次，共{pairs_per_row}次比较/条")
    print(f"配置: 并发={CONCURRENCY}, 超时={REQUEST_TIMEOUT}s")
    
    results = await process_all(df, response_cols, BASE_URL, MODEL_NAME, API_KEY,
                                CONCURRENCY, REQUEST_TIMEOUT, MAX_RETRIES)
    
    # 保存结果 - 基于输入文件名命名
    input_basename = os.path.splitext(os.path.basename(DATA_PATH))[0]
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(OUTPUT_DIR, f'{input_basename}-result-{timestamp}.xlsx')
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
        print("\n最佳Response分布:")
        for col, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {col}: {cnt} ({cnt/success*100:.1f}%)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
