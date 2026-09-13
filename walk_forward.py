import optuna
import yaml
import time
import pandas as pd
from datetime import datetime, timedelta
from backtest import run_backtest, generate_reports

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_objective(train_start, train_end):
    """Factory function to pass specific dates into the Optuna objective."""
    def objective(trial):
        params = load_config()
        
        # Step-based suggestions
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
        
        # Train strictly on the Training Window
        trades_df = run_backtest(params, start_date=train_start, end_date=train_end)
        
        if trades_df.empty or len(trades_df) < 20:
            return -999.0 
            
        trades_df['PnL_Pct'] = (trades_df['Exit'] - trades_df['Entry']) / trades_df['Entry']
        mean_return = trades_df['PnL_Pct'].mean()
        std_return = trades_df['PnL_Pct'].std()
        
        return mean_return / (std_return + 1e-6)
        
    return objective

if __name__ == "__main__":
    # Define our 5-year backtest horizon (2021 to 2026)
    # We will test in 1-year blocks. 
    # For each block, we train on the 5 years prior.
    
    test_blocks = [
        {"train_start": "2016-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
        {"train_start": "2017-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        {"train_start": "2018-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"train_start": "2019-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        {"train_start": "2020-01-01", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2026-12-31"}, # Includes YTD 2026
    ]
    
    all_oos_trades = []
    
    print("Starting Walk-Forward Optimization...")
    
    for idx, block in enumerate(test_blocks):
        print(f"\n--- WFO WINDOW {idx+1}/5 ---")
        print(f"Training Data: {block['train_start']} to {block['train_end']}")
        print(f"Testing Data:  {block['test_start']} to {block['test_end']}")
        
        # 1. Train the ML (6 minutes per window to hit a ~30 min total budget)
        study = optuna.create_study(direction="maximize")
        study.optimize(create_objective(block['train_start'], block['train_end']), timeout=360)
        
        best_params = load_config()
        # Apply the optimized values discovered for this specific window
        for k in best_params['strategy'].keys():
            if k in study.best_params: best_params['strategy'][k] = study.best_params[k]
        for k in best_params['trade_management'].keys():
            if k in study.best_params: best_params['trade_management'][k] = study.best_params[k]
            
        print(f"Best RR for Window {idx+1}: {study.best_params['target_rr_ratio']}")
        
        # 2. Test strictly on Out-of-Sample data using these optimized parameters
        oos_trades_df = run_backtest(best_params, start_date=block['test_start'], end_date=block['test_end'])
        
        if not oos_trades_df.empty:
            all_oos_trades.append(oos_trades_df)
            print(f"Generated {len(oos_trades_df)} out-of-sample trades.")

    # 3. Combine all Out-Of-Sample periods into one master backtest
    if all_oos_trades:
        master_trades_df = pd.concat(all_oos_trades, ignore_index=True)
        print("\n--- FINAL 5-YEAR OUT-OF-SAMPLE RESULTS ---")
        generate_reports(master_trades_df)
    else:
        print("No trades generated across all Out-of-Sample windows.")
