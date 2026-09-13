import optuna
import yaml
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest(params, start_date=None, end_date=None):
    with open("market_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    trades = []
    strat_params = params['strategy']
    mgmt = params['trade_management']
    
    for ticker, df in data.items():
        if ticker == "^CNX100":
            continue
            
        df = df.copy()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=mgmt['ema_stoploss_period'], adjust=False).mean()
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        
        period = int(strat_params['max_consolidation_period'])
        df['Box_High'] = df['High'].rolling(window=period).max()
        df['Box_Low'] = df['Low'].rolling(window=period).min()
        df['Box_Range_Pct'] = ((df['Box_High'] - df['Box_Low']) / df['Box_Low']) * 100
        
        exp_period = int(strat_params['expansion_lookback_period'])
        df['Prev_Expansion'] = (df['Close'] / df['Close'].shift(exp_period) - 1) * 100
        
        in_trade = False
        entry_price, sl, target = 0, 0, 0
        entry_date = None
        
        for i in range(period + exp_period, len(df)):
            if not in_trade:
                if df['Box_Range_Pct'].iloc[i-1] <= strat_params['max_consolidation_range_pct']:
                    if df['Prev_Expansion'].iloc[i-period] >= strat_params['prev_expansion_phase_return_pct']:
                        box_high = df['Box_High'].iloc[i-1]
                        box_low = df['Box_Low'].iloc[i-1]
                        
                        req_close = box_high * (1 + strat_params['min_close_pct_beyond_box']/100)
                        if df['Close'].iloc[i] > req_close:
                            if df['Volume'].iloc[i] > df['Vol_SMA'].iloc[i] * strat_params['min_volume_increase_ratio']:
                                
                                candle_range = df['High'].iloc[i] - df['Low'].iloc[i]
                                wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
                                wick_pct = wick / candle_range if candle_range > 0 else 0
                                
                                if wick_pct <= strat_params['max_wick_length_on_breakout_pct']:
                                    if not strat_params['req_price_beyond_ema20'] or (df['Close'].iloc[i] > df['EMA20'].iloc[i]):
                                        
                                        in_trade = True
                                        entry_price = df['Close'].iloc[i]
                                        
                                        box_sl = box_high - strat_params['stop_position_inside_box'] * (box_high - box_low)
                                        ema200_sl = df['EMA200'].iloc[i]
                                        risk_sl = entry_price * (1 - strat_params['max_sl_risk_pct']/100)
                                        
                                        sl = max(box_sl, ema200_sl, risk_sl)
                                        risk = entry_price - sl
                                        target = entry_price + (risk * mgmt['target_rr_ratio'])
                                        
                                        entry_date = df.index[i]
            else:
                if df['Low'].iloc[i] < sl:
                    trades.append({'Ticker': ticker, 'Entry Date': entry_date, 'Exit Date': df.index[i], 
                                   'Entry': entry_price, 'Exit': sl, 'Result': 'SL'})
                    in_trade = False
                elif df['High'].iloc[i] > target:
                    trades.append({'Ticker': ticker, 'Entry Date': entry_date, 'Exit Date': df.index[i], 
                                   'Entry': entry_price, 'Exit': target, 'Result': 'Target'})
                    in_trade = False
                else:
                    trail_level = df['Close'].iloc[i] * (1 - mgmt['trailing_sl_pct']/100)
                    if trail_level > sl:
                        sl = trail_level

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        if start_date:
            trades_df = trades_df[trades_df['Entry Date'] >= pd.to_datetime(start_date)]
        if end_date:
            trades_df = trades_df[trades_df['Entry Date'] <= pd.to_datetime(end_date)]
            
    return trades_df

def generate_reports(trades_df):
    if trades_df.empty:
        print("No trades generated.")
        return 0

    trades_df['PnL_Pct'] = (trades_df['Exit'] - trades_df['Entry']) / trades_df['Entry'] * 100
    total_return = trades_df['PnL_Pct'].sum()
    win_rate = len(trades_df[trades_df['PnL_Pct'] > 0]) / len(trades_df) * 100
    
    with open("stats.txt", "w") as f:
        f.write(f"Total Trades: {len(trades_df)}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Total Cumulative Return: {total_return:.2f}%\n")
    
    trades_df.to_csv("trades.csv", index=False)
    
    with open("market_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    benchmark = data.get("^CNX100")
    if benchmark is not None:
        benchmark['Cum_Return'] = (benchmark['Close'].pct_change().fillna(0) + 1).cumprod()
        
        plt.figure(figsize=(10,6))
        plt.plot(benchmark.index, benchmark['Cum_Return'], label='Nifty 100 Index', color='gray')
        
        trades_df = trades_df.sort_values('Exit Date')
        trades_df['Cum_Strat'] = (trades_df['PnL_Pct'] / 100 + 1).cumprod()
        plt.plot(trades_df['Exit Date'], trades_df['Cum_Strat'], label='Breakout Strategy (OOS)', color='blue', marker='o', markersize=3)
        
        plt.title('Walk-Forward Strategy vs Index')
        plt.legend()
        plt.savefig("comparison.png")

    return total_return

def create_objective(train_start, train_end):
    def objective(trial):
        params = load_config()
        
        params['strategy']['max_consolidation_range_pct'] = trial.suggest_float('max_consolidation_range_pct', 2.0, 15.0, step=0.5)
        params['strategy']['max_consolidation_period'] = trial.suggest_int('max_consolidation_period', 10, 60, step=2)
        params['strategy']['stop_position_inside_box'] = trial.suggest_float('stop_position_inside_box', 0.1, 0.9, step=0.1)
        params['strategy']['max_sl_risk_pct'] = trial.suggest_float('max_sl_risk_pct', 0.5, 5.0, step=0.25)
        params['strategy']['min_close_pct_beyond_box'] = trial.suggest_float('min_close_pct_beyond_box', 0.25, 3.0, step=0.25)
        params['strategy']['max_wick_length_on_breakout_pct'] = trial.suggest_float('max_wick_length_on_breakout_pct', 0.1, 1.0, step=0.1)
        params['strategy']['min_volume_increase_ratio'] = trial.suggest_float('min_volume_increase_ratio', 1.0, 4.0, step=0.25)
        params['strategy']['req_price_beyond_ema20'] = trial.suggest_categorical('req_price_beyond_ema20', [True, False])
        params['strategy']['prev_expansion_phase_return_pct'] = trial.suggest_float('prev_expansion_phase_return_pct', 5.0, 30.0, step=1.0)
        params['strategy']['expansion_lookback_period'] = trial.suggest_int('expansion_lookback_period', 20, 100, step=5)
        
        params['trade_management']['ema_stoploss_period'] = trial.suggest_int('ema_stoploss_period', 50, 300, step=10)
        params['trade_management']['target_rr_ratio'] = trial.suggest_float('target_rr_ratio', 1.0, 5.0, step=0.25)
        params['trade_management']['trailing_sl_pct'] = trial.suggest_float('trailing_sl_pct', 2.0, 15.0, step=0.5)
        
        trades_df = run_backtest(params, start_date=train_start, end_date=train_end)
        
        if trades_df.empty or len(trades_df) < 20:
            return -999.0 
            
        trades_df['PnL_Pct'] = (trades_df['Exit'] - trades_df['Entry']) / trades_df['Entry']
        mean_return = trades_df['PnL_Pct'].mean()
        std_return = trades_df['PnL_Pct'].std()
        
        return mean_return / (std_return + 1e-6)
        
    return objective

if __name__ == "__main__":
    test_blocks = [
        {"train_start": "2016-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
        {"train_start": "2017-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        {"train_start": "2018-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"train_start": "2019-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        {"train_start": "2020-01-01", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2026-12-31"}, 
    ]
    
    all_oos_trades = []
    print("Starting Deep 5.8-Hour Walk-Forward Optimization...")
    
    best_params = load_config()
    
    for idx, block in enumerate(test_blocks):
        print(f"\n--- WFO WINDOW {idx+1}/5 ---")
        
        study = optuna.create_study(direction="maximize")
        # 4150 seconds per window (~69 mins * 5 windows = ~345 mins / 5.75 hours)
        study.optimize(create_objective(block['train_start'], block['train_end']), timeout=4150)
        
        # Clean and round parameters to completely eliminate trailing float noise
        for k in best_params['strategy'].keys():
            if k in study.best_params:
                val = study.best_params[k]
                best_params['strategy'][k] = round(val, 2) if isinstance(val, float) else val
                
        for k in best_params['trade_management'].keys():
            if k in study.best_params:
                val = study.best_params[k]
                best_params['trade_management'][k] = round(val, 2) if isinstance(val, float) else val
            
        oos_trades_df = run_backtest(best_params, start_date=block['test_start'], end_date=block['test_end'])
        if not oos_trades_df.empty:
            all_oos_trades.append(oos_trades_df)

    if all_oos_trades:
        master_trades_df = pd.concat(all_oos_trades, ignore_index=True)
        print("\n--- FINAL 5-YEAR OUT-OF-SAMPLE RESULTS ---")
        generate_reports(master_trades_df)
        
        with open("config.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
    else:
        print("No trades generated across all Out-of-Sample windows.")
