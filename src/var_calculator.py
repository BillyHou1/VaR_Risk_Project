import numpy as np
import pandas as pd
from scipy import stats

def historical_var(returns, window=252, alpha=0.05):
    return returns.rolling(window).apply(lambda x: np.percentile(x, alpha * 100), raw=True)

def parametric_var(returns, window=252, alpha=0.05):
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    return mu + sigma * stats.norm.ppf(alpha)

def ewma_var(returns, lam=0.94, alpha=0.05):
    # RiskMetrics: sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
    sigma = np.sqrt((returns ** 2).ewm(alpha=1-lam, adjust=False).mean())
    return sigma * stats.norm.ppf(alpha)

def expected_shortfall(returns, window=252, alpha=0.05):
    def es(x):
        v = np.percentile(x, alpha * 100)
        tail = x[x <= v]
        return tail.mean() if len(tail) else v
    return returns.rolling(window).apply(es, raw=True)

def regime_aware_var(returns, regimes, alpha=0.05, min_obs=30):
    # parametric VaR fitted per regime (uses past regime-conditioned mu/sigma)
    out = pd.Series(index=returns.index, dtype=float)
    z = stats.norm.ppf(alpha)
    for t in range(min_obs, len(returns)):
        r_past, g_past = returns.iloc[:t], regimes.iloc[:t]
        cur = regimes.iloc[t]
        sub = r_past[g_past == cur]
        if len(sub) < min_obs:
            sub = r_past
        out.iloc[t] = sub.mean() + sub.std() * z
    return out

def calculate_all_var(returns, window=252, alpha=0.05):
    return pd.DataFrame({
        'returns': returns,
        'hist_var': historical_var(returns, window, alpha),
        'param_var': parametric_var(returns, window, alpha),
        'ewma_var': ewma_var(returns, alpha=alpha),
        'es': expected_shortfall(returns, window, alpha),
    }).dropna()

if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    df = load_spy_data('data/raw/spy_raw.csv')
    feats = calculate_features(df)
    res = calculate_all_var(feats['returns'])
    print(res.tail())
