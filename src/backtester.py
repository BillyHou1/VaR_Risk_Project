# VaR Backtesting
import numpy as np
import pandas as pd
from scipy import stats

# Backtesting function
def backtest(returns, var, alpha=0.05):
    idx = returns.index.intersection(var.index)
    ret = returns.loc[idx]
    v = var.loc[idx]
    violations = ret < v
    n = len(violations)
    n_viol = violations.sum()
    viol_rate = n_viol / n
    print(f"Violations: {n_viol}/{n} ({viol_rate:.2%}), expected: {alpha:.2%}")
    # Kupiec POF test
    kupiec_p = kupiec_test(n, n_viol, alpha)
    print(f"Kupiec p-value: {kupiec_p:.4f} {'PASS' if kupiec_p > 0.05 else 'FAIL'}")
    # Christoffersen test
    chris_p = christoffersen_test(violations.values)
    print(f"Christoffersen p-value: {chris_p:.4f} {'PASS' if chris_p > 0.05 else 'FAIL'}")
    return {'n': n, 'n_viol': n_viol, 'viol_rate': viol_rate, 'kupiec_p': kupiec_p, 'chris_p': chris_p}

#i Kupiec Proportion of Failures Test
def kupiec_test(n, n_viol, alpha):
    if n_viol == 0 or n_viol == n:
        return np.nan
    p = n_viol / n
    lr = -2 * (n_viol * np.log(alpha/p) + (n - n_viol) * np.log((1-alpha)/(1-p)))
    return 1 - stats.chi2.cdf(lr, df=1)
# Christoffersen Independence Test
def christoffersen_test(violations):
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(violations)):
        prev, curr = violations[i-1], violations[i]
        if not prev and not curr: n00 += 1
        elif not prev and curr: n01 += 1
        elif prev and not curr: n10 += 1
        else: n11 += 1
    if n00 + n01 == 0 or n10 + n11 == 0:
        return np.nan
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p == 0 or p == 1: # Avoid log(0)
        return np.nan
    eps = 1e-10
    L0 = n00*np.log(1-p+eps) + n01*np.log(p+eps) + n10*np.log(1-p+eps) + n11*np.log(p+eps)
    L1 = n00*np.log(1-p01+eps) + n01*np.log(p01+eps) + n10*np.log(1-p11+eps) + n11*np.log(p11+eps)
    lr = -2 * (L0 - L1)
    return 1 - stats.chi2.cdf(lr, df=1)

# Example usage
if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features, train_test_split
    from var_calculator import calculate_all_var
    df = load_spy_data()
    features = calculate_features(df)
    train, test = train_test_split(features)
    var_results = calculate_all_var(features['returns'])
    test_var = var_results.loc[test.index]
    print("\nHistorical VaR:")
    backtest(test['returns'], test_var['hist_var'])
    print("\nParametric VaR:")
    backtest(test['returns'], test_var['param_var'])
    print("\nEWMA VaR:")
    backtest(test['returns'], test_var['ewma_var'])
