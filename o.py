# -*- coding: utf-8 -*-
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 稳定币黑名单，遇到直接跳过不扫描
STABLE_COINS = {'USDC', 'USDG', 'DAI', 'BUSD', 'EUR'}

# ==========================================
# 1. 核心技术指标计算 (标准 V5 算法)
# ==========================================
def calculate_sma(values, n, m=1):
    if len(values) < n: return np.full(len(values), np.nan)
    mask = ~np.isnan(values)
    if not np.any(mask): return np.full(len(values), np.nan)
    start_idx = np.where(mask)[0][0]
    actual_values = values[start_idx:]
    result = np.full(len(values), np.nan)
    if len(actual_values) >= n:
        result[start_idx + n - 1] = np.mean(actual_values[:n])
        for i in range(start_idx + n, len(values)):
            if not np.isnan(values[i]) and not np.isnan(result[i-1]):
                result[i] = (values[i] * m + result[i-1] * (n - m)) / n
    return result

def calculate_ema(values, n):
    if len(values) < n: return np.full(len(values), np.nan)
    mask = ~np.isnan(values)
    if not np.any(mask): return np.full(len(values), np.nan)
    start_idx = np.where(mask)[0][0]
    actual_values = values[start_idx:]
    result = np.full(len(values), np.nan)
    if len(actual_values) >= n:
        result[start_idx + n - 1] = np.mean(actual_values[:n])
        for i in range(start_idx + n, len(values)):
            if not np.isnan(values[i]) and not np.isnan(result[i-1]):
                result[i] = (values[i] * 2 + result[i-1] * (n - 1)) / (n + 1)
    return result

# ==========================================
# 2. V5 标准版扫描逻辑
# ==========================================
def check_crypto_conditions_v5(df):
    vals = df['close'].values
    if len(vals) < 500: return False
    
    sma13_arr = calculate_sma(vals, 13, 1)
    sma55_arr = calculate_sma(vals, 55, 1)
    ema144_arr = calculate_ema(vals, 144)
    ema169_arr = calculate_ema(vals, 169)
    
    sma13 = pd.Series(sma13_arr, index=df.index)
    sma55 = pd.Series(sma55_arr, index=df.index)

    diff_ratio = (sma13 - sma55).abs() / np.minimum(sma13, sma55)
    is_close = diff_ratio <= 0.025
    
    change_points = is_close.ne(is_close.shift()).cumsum()
    close_groups = [g for _, g in is_close[is_close].groupby(change_points[is_close])]
    
    if len(close_groups) < 2: return False
    
    group_b = close_groups[-1]
    group_a = close_groups[-2]
    
    idx_a_start = df.index.get_indexer([group_a.index[0]])[0]
    idx_a_end = df.index.get_indexer([group_a.index[-1]])[0]
    idx_b_start = df.index.get_indexer([group_b.index[0]])[0]
    
    if not is_close.iloc[-1] or sma13.iloc[-1] <= sma13.iloc[-2]: return False
    
    if idx_a_end == -1 or idx_b_start == -1: return False
    between_zone = sma13.iloc[idx_a_end + 1 : idx_b_start]
    if len(between_zone) > 0:
        if not (between_zone > sma55.iloc[idx_a_end + 1 : idx_b_start]).all():
            return False
    
    idx_pre_a = idx_a_start - 1
    if idx_pre_a < 0: return False
    pre_a_ratio = (sma13.iloc[idx_pre_a] - sma55.iloc[idx_pre_a]) / sma55.iloc[idx_pre_a]
    if pre_a_ratio <= 0.025: return False 
    
    if pd.isnull(ema144_arr[-1]) or ema144_arr[-1] <= ema169_arr[-1]: return False
        
    return True

# ==========================================
# 3. OKX 数据获取与智能 CCY 去重 (永续合约优先 + 稳定币过滤)
# ==========================================
def get_okx_tickers_optimized():
    spot_list = []
    swap_list = []
    url = "https://www.okx.com/api/v5/market/tickers"
    
    try:
        res_spot = requests.get(f"{url}?instType=SPOT", timeout=12).json()
        if res_spot.get('code') == '0' and isinstance(res_spot.get('data'), list):
            for item in res_spot['data']:
                inst_id = item.get('instId', '')
                base_ccy = inst_id.split('-')[0]
                # 过滤稳定币
                if base_ccy in STABLE_COINS: continue
                
                vol_usdt = float(item.get('volCcy24h', 0))
                if inst_id.endswith('-USDT') and vol_usdt >= 30000000:
                    spot_list.append(inst_id)
    except Exception as e:
        print(f"获取 SPOT 列表异常: {e}")

    try:
        res_swap = requests.get(f"{url}?instType=SWAP", timeout=12).json()
        if res_swap.get('code') == '0' and isinstance(res_swap.get('data'), list):
            for item in res_swap['data']:
                inst_id = item.get('instId', '')
                base_ccy = inst_id.split('-')[0]
                # 过滤稳定币
                if base_ccy in STABLE_COINS: continue
                
                vol_usdt = float(item.get('volCcy24h', 0))
                if inst_id.endswith('-USDT-SWAP') and vol_usdt >= 30000000:
                    swap_list.append(inst_id)
    except Exception as e:
        print(f"获取 SWAP 列表异常: {e}")

    final_targets = []
    swap_ccy_set = set()
    
    for inst in swap_list:
        base_ccy = inst.split('-')[0]
        swap_ccy_set.add(base_ccy)
        final_targets.append((inst, 'swap'))
        
    for inst in spot_list:
        base_ccy = inst.split('-')[0]
        if base_ccy not in swap_ccy_set:
            final_targets.append((inst, 'spot'))

    return sorted(final_targets, key=lambda x: x[0])

def get_okx_candles(inst_id, m_type, okx_bar, limit=1000):
    url = "https://www.okx.com/api/v5/market/history-candles"
    all_candles = []
    after_ts = ""
    
    # 💡 核心修复：如果是合约(swap)，需要把 -SWAP 去掉，K线接口才能正确识别 BTC-USDT 资产
    req_inst_id = inst_id.replace('-SWAP', '') if m_type == 'swap' else inst_id
    
    for _ in range(10):
        req_url = f"{url}?instId={req_inst_id}&bar={okx_bar}&limit=100"
        if after_ts: 
            req_url += f"&after={after_ts}"
        try:
            res = requests.get(req_url, timeout=12).json()
            if res.get('code') == '0' and isinstance(res.get('data'), list) and len(res['data']) > 0:
                data_chunk = res['data']
                all_candles.extend(data_chunk)
                after_ts = data_chunk[-1][0]
                if len(data_chunk) < 100: 
                    break
            else: 
                break
        except Exception: 
            break
        time.sleep(0.01)

    if len(all_candles) == 0: 
        return pd.DataFrame()
        
    df = pd.DataFrame(all_candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
    df = df[df['confirm'] == '1']
    df = df.iloc[::-1]
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df.set_index('ts', inplace=True)
    df['close'] = df['close'].astype(float)
    return df

# ==========================================
# 4. 常规白底黑字 HTML 报告生成
# ==========================================
def generate_html_report(category_results, start_str, end_str, duration_str, no_data=False):
    total_hits = sum(len(v) for v in category_results.values()) if not no_data else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ background-color: #ffffff; color: #333333; font-family: Arial, sans-serif; padding: 10px; }}
            .report-title {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
            .meta-info {{ font-size: 13px; color: #666666; margin-bottom: 20px; line-height: 1.5; }}
            .section-head {{ font-size: 14px; font-weight: bold; margin-top: 15px; margin-bottom: 5px; border-bottom: 1px solid #cccccc; padding-bottom: 3px; }}
            .list-item {{ font-size: 13px; margin-bottom: 4px; font-family: monospace; }}
            .empty-msg {{ font-size: 13px; color: #999999; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="report-title">OKX V5 盘中扫描报告</div>
        <div class="meta-info">
            开始时间：{start_str}<br>
            结束时间：{end_str}<br>
            核心总计耗时：{duration_str}<br>
            当前命中总数：{total_hits} 个
        </div>
    """
    
    if no_data:
        html += '<div class="empty-msg" style="color: #cc0000; font-weight: bold;">⚠️ 提示：未获取到当前符合交易量要求的活跃资产，本次未扫描。</div>'
    else:
        for freq in ['1d', '4h', '2h', '1h']:
            html += f'<div class="section-head">【{freq} 周期频率】</div>'
            stocks = category_results.get(freq, [])
            if stocks:
                for stock in stocks:
                    html += f'<div class="list-item"> - {stock}</div>'
            else:
                html += '<div class="empty-msg"> 暂无匹配标的</div>'
            
    html += "</body></html>"
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    with open("subject.txt", "w", encoding="utf-8") as f:
        if no_data:
            f.write("OKX V5 扫描报告: 暂无活跃标的数据")
        else:
            f.write(f"OKX V5 扫描报告: 发现 {total_hits} 个潜在机会")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    start_time = datetime.now()
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print("="*60)
    print(f"▶️ OKX 智能去重雷达启动: {start_str}")
    target_pairs = get_okx_tickers_optimized()
    print(f"合并去重完毕（已剔除稳定币），共有 {len(target_pairs)} 个独立标的进入扫描序列。")
    print("="*60)

    if not target_pairs:
        end_time = datetime.now()
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        duration = end_time - start_time
        duration_str = f"{int(duration.total_seconds() // 60)}分{int(duration.total_seconds() % 60)}秒"
        generate_html_report({}, start_str, end_str, duration_str, no_data=True)
        return

    category_results = {'1d': [], '4h': [], '2h': [], '1h': []}
    timeframes = [('1d', '1DUTC'), ('4h', '4H'), ('2h', '2H'), ('1h', '1H')]

    for i, (inst_id, m_type) in enumerate(target_pairs):
        display_name = f"{inst_id} ({m_type.upper()})"
        print(f"[{i+1}/{len(target_pairs)}] 正在扫描: {display_name} ...", end="\r")
        
        for freq_name, okx_bar in timeframes:
            # 💡 这里将 m_type 传了进去，内部会自动处理合约的 K 线名称
            df = get_okx_candles(inst_id, m_type, okx_bar, limit=1000)
            if not df.empty:
                if check_crypto_conditions_v5(df):
                    category_results[freq_name].append(display_name)
                    print(f"\n🔥 [OKX 信号] 周期: {freq_name} -> {display_name}")
                    
            time.sleep(0.01) 

    end_time = datetime.now()
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    duration = end_time - start_time
    duration_str = f"{int(duration.total_seconds() // 60)}分{int(duration.total_seconds() % 60)}秒"

    generate_html_report(category_results, start_str, end_str, duration_str)
    print("\n[系统提示] HTML 结果文件与主题已就绪。")

if __name__ == "__main__":
    main()
