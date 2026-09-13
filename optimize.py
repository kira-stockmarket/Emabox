import optuna
import yaml
import time
from backtest import run_backtest, generate_reports

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def objective(trial):
    params = load_config()
    
    # --- STRATEGY PARAMETERS ---
    params['strategy']['max_consolidation_range_pct'] = trial.suggest_float('max_consolidation_range_pct', 2.0, 15.0)
    params['strategy']['max_consolidation_period'] = trial.suggest_int('max_consolidation_period', 10, 60)
    params['strategy']['stop_position_inside_box'] = trial.suggest_float('stop_position_inside_box', 0.1, 0.9)
    params['strategy']['max_sl_risk_pct'] = trial.suggest_float('max_sl_risk_pct', 0.5, 5.0)
    params['strategy']['min_close_pct_beyond_box'] = trial.suggest_float('min_close_pct_beyond_box', 0.1, 3.0)
    params['strategy']['max_wick_length_on_breakout_pct'] = trial.suggest_float('max_wick_length_on_breakout_pct', 0.1, 1.0)
    params['strategy']['min_volume_increase_ratio'] = trial.suggest_float('min_volume_increase_ratio', 1.0, 4.0)
    params['strategy']['req_price_beyond_ema20'] = trial.suggest_categorical('req_price_beyond_ema20', [True, False])
    params['strategy']['prev_expansion_phase_return_pct'] = trial.suggest_float('prev_expansion_phase_return_pct', 5.0, 30.0)
    params['strategy']['expansion_lookback_period'] = trial.suggest_int('expansion_lookback_period', 20, 100)
    
    # --- TRADE MANAGEMENT PARAMETERS ---
    params['trade_management']['ema_stoploss_period'] = trial.suggest_int('ema_stoploss_period', 50, 300)
    params['trade_management']['target_rr_ratio'] = trial.suggest_float('target_rr_ratio', 1.0, 5.0)
    params['trade_management']['trailing_sl_pct'] = trial.suggest_float('trailing_sl_pct', 2.0, 15.0)
    
    # Run backtest with these trial parameters
    trades_df = run_backtest(params)
    
    # Penalize parameter sets that generate too few trades (overfitting prevention)
    if trades_df.empty or len(trades_df) < 5:
        return -999.0 
        
    trades_df['PnL_Pct'] = (trades_df['Exit'] - trades_df['Entry']) / trades_df['Entry']
    
    # Objective: Maximize Sharpe Ratio (Mean Return / Std Dev)
    mean_return = trades_df['PnL_Pct'].mean()
    std_return = trades_df['PnL_Pct'].std()
    
    if std_return == 0: 
        return 0
        
    return mean_return / std_return 

if __name__ == "__main__":
    print("Starting Comprehensive ML Parameter Optimization (30 Minutes)...")
    
    # Create study optimizing for maximum return/risk ratio
    study = optuna.create_study(direction="maximize")
    
    # Run for 1800 seconds (30 minutes)
    study.optimize(objective, timeout=600)
    
    print("\nBest Parameters found:")
    best = study.best_params
    for key, value in best.items():
        print(f"  {key}: {value}")
    
    # Load original config to overwrite with best params
    config = load_config()
    
    # Save Strategy Params 
    # (Converting Optuna types strictly to native Python types for YAML serialization)
    config['strategy']['max_consolidation_range_pct'] = float(round(best['max_consolidation_range_pct'], 2))
    config['strategy']['max_consolidation_period'] = int(best['max_consolidation_period'])
    config['strategy']['stop_position_inside_box'] = float(round(best['stop_position_inside_box'], 2))
    config['strategy']['max_sl_risk_pct'] = float(round(best['max_sl_risk_pct'], 2))
    config['strategy']['min_close_pct_beyond_box'] = float(round(best['min_close_pct_beyond_box'], 2))
    config['strategy']['max_wick_length_on_breakout_pct'] = float(round(best['max_wick_length_on_breakout_pct'], 2))
    config['strategy']['min_volume_increase_ratio'] = float(round(best['min_volume_increase_ratio'], 2))
    config['strategy']['req_price_beyond_ema20'] = bool(best['req_price_beyond_ema20'])
    config['strategy']['prev_expansion_phase_return_pct'] = float(round(best['prev_expansion_phase_return_pct'], 2))
    config['strategy']['expansion_lookback_period'] = int(best['expansion_lookback_period'])
    
    # Save Management Params
    config['trade_management']['ema_stoploss_period'] = int(best['ema_stoploss_period'])
    config['trade_management']['target_rr_ratio'] = float(round(best['target_rr_ratio'], 2))
    config['trade_management']['trailing_sl_pct'] = float(round(best['trailing_sl_pct'], 2))
    
    # Overwrite the config file so the final backtest run uses these optimized parameters
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("\nSuccessfully updated config.yaml with best parameters.")
