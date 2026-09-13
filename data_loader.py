import yfinance as yf
import pandas as pd
import pickle
import os

# Sample of highly liquid Nifty 100 stocks (add full 100 list as needed)
NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS"
]
BENCHMARK = "^CNX100" # Nifty 100 Index

def download_data():
    print("Downloading Nifty 100 historical data...")
    all_tickers = NIFTY_TICKERS + [BENCHMARK]
    
    # Download last 5 years of daily data
    data = yf.download(all_tickers, period="5y", group_by="ticker", auto_adjust=True)
    
    clean_data = {}
    for ticker in all_tickers:
        try:
            df = data[ticker].dropna()
            if len(df) > 200:
                clean_data[ticker] = df
        except KeyError:
            continue
            
    with open("market_data.pkl", "wb") as f:
        pickle.dump(clean_data, f)
    print(f"Data saved for {len(clean_data)} tickers.")

if __name__ == "__main__":
    download_data()
