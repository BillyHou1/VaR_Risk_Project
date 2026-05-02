import numpy as np
import pandas as pd

def _regime_params(returns, regimes):
    # mu, sigma per regime label
    return {g: (sub.mean(), sub.std()) for g, sub in returns.groupby(regimes) if len(sub) > 5}

def monte_carlo_paths(returns, regimes=None, current_regime=None,
                      horizon=20, n_paths=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    if regimes is not None and current_regime is not None:
        params = _regime_params(returns, regimes)
        mu, sigma = params.get(current_regime, (returns.mean(), returns.std()))
        cond = f"regime={current_regime}"
    else:
        mu, sigma = returns.mean(), returns.std()
        cond = "unconditional"
    shocks = rng.normal(mu, sigma, size=(n_paths, horizon))
    cum_paths = (1 + shocks).cumprod(axis=1)
    final = cum_paths[:, -1] - 1.0   # cumulative return at horizon
    var_h = np.percentile(final, alpha * 100)
    es_h = final[final <= var_h].mean() if (final <= var_h).any() else var_h
    print(f"MC ({cond}): paths={n_paths} h={horizon}d  mu={mu:.5f} sigma={sigma:.5f}")
    print(f"  H-day VaR{int((1-alpha)*100)} = {var_h:.4f}   ES = {es_h:.4f}")
    return {'paths': cum_paths, 'final': final, 'var_h': var_h, 'es_h': es_h,
            'mu': mu, 'sigma': sigma, 'horizon': horizon, 'condition': cond}

def forecast_all_regimes(returns, regimes, horizon=20, n_paths=10000, alpha=0.05, seed=42):
    out = {}
    for g in sorted(regimes.dropna().unique()):
        out[g] = monte_carlo_paths(returns, regimes, g, horizon, n_paths, alpha, seed)
    return out

if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    from gmm_risk_states import predict_risk_states
    df = load_spy_data('data/raw/spy_raw.csv')
    feats = calculate_features(df)
    states, *_ = predict_risk_states(feats)
    aligned = feats.loc[states.index]
    cur = states['state_name'].iloc[-1]
    monte_carlo_paths(aligned['returns'], states['state_name'], cur)
