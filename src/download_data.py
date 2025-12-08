# Download SPY Data
import os
import pandas as pd
import yfinance as yf
# Download SPY data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# Ensure data/raw directory exists
def download_spy(start='2010-01-01', end='2025-11-27'):
    print(f"Downloading SPY from {start} to {end}...")
    data = yf.download('SPY', start=start, end=end, progress=True)
    if data.empty:
        raise ValueError("No data downloaded")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"Downloaded {len(data)} trading days")
    raw_path = os.path.join(DATA_DIR, 'raw', 'spy_raw.csv')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    data.to_csv(raw_path)
    print(f"Saved: {raw_path}")
    return data

# Example usage
if __name__ == "__main__":
    data = download_spy()
    print(data.tail())
