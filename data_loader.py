import yfinance as yf
import pandas as pd
import pickle

# Top 50 highly liquid Nifty stocks
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "INFY.NS", "ITC.NS", 
    "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS", 
    "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS", "TATAMOTORS.NS", "NTPC.NS", "AXISBANK.NS", "DMART.NS", 
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", 
    "BAJAJ-AUTO.NS", "POWERGRID.NS", "NESTLEIND.NS", "WIPRO.NS", "M&M.NS", "IOC.NS", "HAL.NS", "DLF.NS", 
    "TATASTEEL.NS", "SIEMENS.NS", "VBL.NS", "IRFC.NS", "GRASIM.NS", "SBILIFE.NS", "BEL.NS", "LTIM.NS", 
    "TRENT.NS", "PNB.NS", "INDIGO.NS", "DRREDDY.NS", "INDUSINDBK.NS", "EICHERMOT.NS"
]

BENCHMARK = "^CNX100" 

def download_data():
    print(f"Downloading MAX historical data for {len(NIFTY_50)} Nifty 50 stocks...")
    all_tickers = NIFTY_50 + [BENCHMARK]
    
    # Using 'max' to get decades of data for robust training
    data = yf.download(all_tickers, period="max", group_by="ticker", auto_adjust=True)
    
    clean_data = {}
    for ticker in all_tickers:
        try:
            df = data[ticker].dropna()
            if len(df) > 500: # Need enough data for rolling windows
                clean_data[ticker] = df
        except KeyError:
            continue
            
    with open("market_data.pkl", "wb") as f:
        pickle.dump(clean_data, f)
    print(f"Data saved for {len(clean_data)} tickers.")

if __name__ == "__main__":
    download_data()
