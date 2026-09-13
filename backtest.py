import pandas as pd
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest(params=None):
    if params is None:
        params = load_config()
        
    with open("market_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    trades = []
    strat_params = params['strategy']
    mgmt = params['trade_management']
    
    for ticker, df in data.items():
        if ticker == "^CNX100":
            continue
            
        df = df.copy()
        
        # Calculate Indicators
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=mgmt['ema_stoploss_period'], adjust=False).mean()
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        
        # Consolidation Box & Expansion Phase
        period = int(strat_params['max_consolidation_period'])
        df['Box_High'] = df['High'].rolling(window=period).max()
        df['Box_Low'] = df['Low'].rolling(window=period).min()
        df['Box_Range_Pct'] = ((df['Box_High'] - df['Box_Low']) / df['Box_Low']) * 100
        
        exp_period = int(strat_params['expansion_lookback_period'])
        df['Prev_Expansion'] = (df['Close'] / df['Close'].shift(exp_period) - 1) * 100
        
        in_trade = False
        entry_price, sl, target = 0, 0, 0
        
        for i in range(period + exp_period, len(df)):
            if not in_trade:
                # 1. Box Check
                if df['Box_Range_Pct'].iloc[i-1] <= strat_params['max_consolidation_range_pct']:
                    # 2. Expansion Phase Check
                    if df['Prev_Expansion'].iloc[i-period] >= strat_params['prev_expansion_phase_return_pct']:
                        box_high = df['Box_High'].iloc[i-1]
                        box_low = df['Box_Low'].iloc[i-1]
                        
                        # 3. Breakout Conditions
                        req_close = box_high * (1 + strat_params['min_close_pct_beyond_box']/100)
                        if df['Close'].iloc[i] > req_close:
                            
                            # 4. Volume Check
                            if df['Volume'].iloc[i] > df['Vol_SMA'].iloc[i] * strat_params['min_volume_increase_ratio']:
                                
                                # 5. Wick Check
                                candle_range = df['High'].iloc[i] - df['Low'].iloc[i]
                                wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
                                wick_pct = wick / candle_range if candle_range > 0 else 0
                                
                                if wick_pct <= strat_params['max_wick_length_on_breakout_pct']:
                                    
                                    # 6. EMA 20 Check
                                    if not strat_params['req_price_beyond_ema20'] or (df['Close'].iloc[i] > df['EMA20'].iloc[i]):
                                        
                                        # Entry Setup
                                        in_trade = True
                                        entry_price = df['Close'].iloc[i]
                                        
                                        # Stoploss Calculation
                                        box_sl = box_high - strat_params['stop_position_inside_box'] * (box_high - box_low)
                                        ema200_sl = df['EMA200'].iloc[i]
                                        risk_sl = entry_price * (1 - strat_params['max_sl_risk_pct']/100)
                                        
                                        sl = max(box_sl, ema200_sl, risk_sl)
                                        risk = entry_price - sl
                                        target = entry_price + (risk * mgmt['target_rr_ratio'])
                                        
                                        entry_date = df.index[i]
            else:
                # Manage Open Position (Trailing SL & Target)
                if df['Low'].iloc[i] < sl:
                    trades.append({'Ticker': ticker, 'Entry Date': entry_date, 'Exit Date': df.index[i], 
                                   'Entry': entry_price, 'Exit': sl, 'Result': 'SL'})
                    in_trade = False
                elif df['High'].iloc[i] > target:
                    trades.append({'Ticker': ticker, 'Entry Date': entry_date, 'Exit Date': df.index[i], 
                                   'Entry': entry_price, 'Exit': target, 'Result': 'Target'})
                    in_trade = False
                else:
                    # Trailing SL update based on peak
                    trail_level = df['Close'].iloc[i] * (1 - mgmt['trailing_sl_pct']/100)
                    if trail_level > sl:
                        sl = trail_level

    trades_df = pd.DataFrame(trades)
    return trades_df

def generate_reports(trades_df):
    if trades_df.empty:
        print("No trades generated.")
        return 0

    trades_df['PnL_Pct'] = (trades_df['Exit'] - trades_df['Entry']) / trades_df['Entry'] * 100
    total_return = trades_df['PnL_Pct'].sum()
    win_rate = len(trades_df[trades_df['PnL_Pct'] > 0]) / len(trades_df) * 100
    
    # Write Stats
    with open("stats.txt", "w") as f:
        f.write(f"Total Trades: {len(trades_df)}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Total Cumulative Return: {total_return:.2f}%\n")
    
    trades_df.to_csv("trades.csv", index=False)
    
    # Plotting vs Benchmark
    with open("market_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    benchmark = data.get("^CNX100")
    if benchmark is not None:
        benchmark['Cum_Return'] = (benchmark['Close'].pct_change().fillna(0) + 1).cumprod()
        
        plt.figure(figsize=(10,6))
        plt.plot(benchmark.index, benchmark['Cum_Return'], label='Nifty 100 Index', color='gray')
        
        # Rough estimation of strategy curve (cumulative addition of PnL)
        trades_df['Cum_Strat'] = (trades_df['PnL_Pct'] / 100 + 1).cumprod()
        plt.plot(trades_df['Exit Date'], trades_df['Cum_Strat'], label='Breakout Strategy', color='blue', marker='o')
        
        plt.title('Strategy vs Nifty 100 Index')
        plt.legend()
        plt.savefig("comparison.png")

    return total_return

if __name__ == "__main__":
    df = run_backtest()
    generate_reports(df)
