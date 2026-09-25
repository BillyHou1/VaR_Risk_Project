import numpy as np
import pandas as pd
from scipy import stats

def kupiec_test(n, n_viol, alpha):
    if n_viol == 0 or n_viol == n:
        return np.nan
    p = n_viol / n
    lr = -2 * (n_viol * np.log(alpha/p) + (n - n_viol) * np.log((1-alpha)/(1-p)))
    return 1 - stats.chi2.cdf(lr, df=1)

def christoffersen_test(violations):
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(violations)):
        prev, curr = violations[i-1], violations[i]
        if not prev and not curr: n00 += 1
        elif not prev and curr:   n01 += 1
        elif prev and not curr:   n10 += 1
        else:                     n11 += 1
    if n00 + n01 == 0 or n10 + n11 == 0:
        return np.nan
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0, 1):
        return np.nan
    eps = 1e-10
    L0 = n00*np.log(1-p+eps) + n01*np.log(p+eps) + n10*np.log(1-p+eps) + n11*np.log(p+eps)
    L1 = n00*np.log(1-p01+eps) + n01*np.log(p01+eps) + n10*np.log(1-p11+eps) + n11*np.log(p11+eps)
    return 1 - stats.chi2.cdf(-2 * (L0 - L1), df=1)

def backtest(returns, var, alpha=0.05, label=""):
    idx = returns.index.intersection(var.index)
    ret, v = returns.loc[idx].dropna(), var.loc[idx].dropna()
    idx = ret.index.intersection(v.index)
    ret, v = ret.loc[idx], v.loc[idx]
    violations = (ret < v)
    n, n_viol = len(violations), int(violations.sum())
    rate = n_viol / n if n else 0
    kp = kupiec_test(n, n_viol, alpha)
    cp = christoffersen_test(violations.values)
    tag = f"[{label}] " if label else ""
    print(f"{tag}n={n}  viol={n_viol} ({rate:.2%}, exp={alpha:.2%})  "
          f"Kupiec p={kp:.3f}{'PASS' if not np.isnan(kp) and kp > 0.05 else 'FAIL':>5s}  "
          f"Christoffersen p={cp:.3f}{'PASS' if not np.isnan(cp) and cp > 0.05 else 'FAIL':>5s}")
    return {'label': label, 'n': n, 'n_viol': n_viol, 'viol_rate': rate,
            'kupiec_p': kp, 'chris_p': cp}

def run_all(returns, var_df, alpha=0.05, methods=('hist_var', 'param_var', 'ewma_var')):
    rows = [backtest(returns, var_df[m], alpha, m) for m in methods if m in var_df.columns]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features, train_test_split
    from var_calculator import calculate_all_var
    df = load_spy_data('data/raw/spy_raw.csv')
    feats = calculate_features(df)
    train, test = train_test_split(feats)
    var_res = calculate_all_var(feats['returns'])
    print(run_all(test['returns'], var_res.loc[test.index]))