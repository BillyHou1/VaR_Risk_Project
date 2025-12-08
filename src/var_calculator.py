# VaR Calculator

import numpy as np
import pandas as pd
from scipy import stats

def historical_var(returns, window=252, alpha=0.05):
    return returns.rolling(window).apply(lambda x: np.percentile(x, alpha * 100))

def parametric_var(returns, window=252, alpha=0.05):
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    z = stats.norm.ppf(alpha)
    return mu + sigma * z

def ewma_var(returns, lam=0.94, alpha=0.05):
    # RiskMetrics: sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
    ewma_var = (returns ** 2).ewm(alpha=1-lam, adjust=False).mean()
    sigma = np.sqrt(ewma_var)
    z = stats.norm.ppf(alpha)
    return sigma * z  # mean=0 per RiskMetrics

def expected_shortfall(returns, window=252, alpha=0.05):
    def calc_es(x):
        var = np.percentile(x, alpha * 100)
        tail = x[x <= var]
        return tail.mean() if len(tail) > 0 else var
    return returns.rolling(window).apply(calc_es)

def calculate_all_var(returns, window=252, alpha=0.05):
    results = pd.DataFrame({
        'returns': returns,
        'hist_var': historical_var(returns, window, alpha),
        'param_var': parametric_var(returns, window, alpha),
        'ewma_var': ewma_var(returns, alpha=alpha),
        'es': expected_shortfall(returns, window, alpha)
    })
    return results.dropna()


if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    df = load_spy_data()
    features = calculate_features(df)
    results = calculate_all_var(features['returns'])
    print("VaR Results (last 5 days):")
    print(results.tail())
    print(f"\nAvg VaR (95%):")
    print(f"  Historical: {results['hist_var'].mean():.4f}")
    print(f"  Parametric: {results['param_var'].mean():.4f}")
    print(f"  EWMA:       {results['ewma_var'].mean():.4f}")
