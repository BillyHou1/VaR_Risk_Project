import os
import numpy as np
import pandas as pd
import yfinance as yf

def download_spy(ticker='SPY', start='2010-01-01', end='2025-11-27', save_dir='data/raw'):
    print(f"Downloading {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=True, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, f'{ticker.lower()}_raw.csv')
    data.to_csv(raw_path)
    print(f"Saved: {raw_path}  ({len(data)} rows)")
    return data, raw_path

def load_spy_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date').sort_index()
    df['returns'] = df['Close'].pct_change()
    print(f"Loaded {len(df)} rows: {df.index[0].date()} to {df.index[-1].date()}")
    return df

def calculate_features(df, window=20):
    f = pd.DataFrame(index=df.index)
    f['returns'] = df['returns']
    f['close'] = df['Close']
    if 'Volume' in df.columns:
        f['volume'] = df['Volume']
    f['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    def downside_std(x):
        neg = x[x < 0]
        return neg.std() * np.sqrt(252) if len(neg) >= 2 else np.nan
    f['downside_vol'] = df['returns'].rolling(window).apply(downside_std, raw=False)
    f['skew'] = df['returns'].rolling(window).skew()
    f['kurt'] = df['returns'].rolling(window).kurt()
    # don't drop on missing volume — only require core return-derived features
    core = ['returns', 'volatility', 'downside_vol', 'skew', 'kurt']
    return f.dropna(subset=core)

def train_test_split(df, test_start='2023-01-01'):
    test_start = pd.Timestamp(test_start)
    train, test = df[df.index < test_start], df[df.index >= test_start]
    print(f"Train: {len(train)}, Test: {len(test)}  (split={test_start.date()})")
    return train, test

if __name__ == "__main__":
    data, path = download_spy()
    df = load_spy_data(path)
    feats = calculate_features(df)
    train, test = train_test_split(feats)
