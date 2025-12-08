# Data Loader

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
# Load SPY data
def load_spy_data(filepath=None):
    if filepath is None:
        filepath = DATA_DIR / "raw" / "spy_raw.csv"
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    df['returns'] = df['Close'].pct_change()
    print(f"Loaded {len(df)} rows: {df.index[0].date()} to {df.index[-1].date()}")
    return df
# Calculate features
def calculate_features(df, window=20):
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['returns']
    features['close'] = df['Close']
    features['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    # Download volatility, skewness, kurtosis
    def downside_std(x):
        neg = x[x < 0]
        return neg.std() * np.sqrt(252) if len(neg) >= 2 else np.nan
    features['downside_vol'] = df['returns'].rolling(window).apply(downside_std)
    features['skew'] = df['returns'].rolling(window).skew()
    features['kurt'] = df['returns'].rolling(window).kurt()
    return features.dropna()
# Train test split
def train_test_split(df, test_start='2023-01-01'):
    test_start = pd.Timestamp(test_start)
    train = df[df.index < test_start]
    test = df[df.index >= test_start]
    print(f"Train: {len(train)}, Test: {len(test)}")
    return train, test

# Example usage
if __name__ == "__main__":
    df = load_spy_data()
    features = calculate_features(df)
    train, test = train_test_split(features)
    print(f"Features: {list(features.columns)}")
    print(features.tail())
