import os
import pickle
import numpy as np
import pandas as pd
from var_calculator import historical_var, parametric_var, ewma_var, expected_shortfall
from gmm_risk_states import predict_risk_states, GMM_FEATS
from data_loader import calculate_features

DEFAULT_GMM_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'gmm_bundle.pkl')

def fit_and_save_gmm(features, n_states=3, save_path=DEFAULT_GMM_PATH):
    states_df, gmm, scaler, names = predict_risk_states(features, n_states=n_states)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bundle = {'gmm': gmm, 'scaler': scaler, 'names': names,
              'features': GMM_FEATS, 'n_states': n_states}
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"GMM bundle saved: {save_path}")
    return states_df, bundle

def load_gmm_bundle(path=DEFAULT_GMM_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _smooth_probs(p, alpha=0.02):
    p = np.asarray(p, dtype=float)
    k = len(p)
    return (p + alpha) / (1.0 + k * alpha)

def predict_regime(features_row_or_df, bundle=None, path=DEFAULT_GMM_PATH, smooth=True):
    bundle = bundle or load_gmm_bundle(path)
    if isinstance(features_row_or_df, dict):
        X = np.array([[features_row_or_df[c] for c in bundle['features']]])
    elif isinstance(features_row_or_df, pd.Series):
        X = features_row_or_df[bundle['features']].values.reshape(1, -1)
    else:
        X = features_row_or_df[bundle['features']].values
    Xs = bundle['scaler'].transform(X)
    labels = bundle['gmm'].predict(Xs)
    raw = bundle['gmm'].predict_proba(Xs)
    probs = np.array([_smooth_probs(raw[i]) for i in range(raw.shape[0])]) if smooth else raw
    try:
        loglik = float(bundle['gmm'].score(Xs))
    except Exception:
        loglik = None
    out = []
    for i, lab in enumerate(labels):
        out.append({'state': int(lab),
                    'state_name': bundle['names'][int(lab)],
                    'probs': {bundle['names'][s]: float(probs[i, s])
                              for s in range(bundle['n_states'])},
                    'log_likelihood': loglik})
    return out[0] if X.shape[0] == 1 else out

def compute_var_suite(returns, window=500, alpha=0.05, lam=0.94):
    return pd.DataFrame({
        'returns': returns,
        'hist_var': historical_var(returns, window, alpha),
        'param_var': parametric_var(returns, window, alpha),
        'ewma_var': ewma_var(returns, lam=lam, alpha=alpha),
        'es': expected_shortfall(returns, window, alpha),
    }).dropna()

def predict_var_latest(close, window=500, alpha=0.05, gmm_path=None):
    returns = close.pct_change().dropna()
    if len(returns) < window:
        window = max(60, len(returns) // 2)
        print(f"[predict_var_latest] short series ({len(returns)} obs); shrinking window to {window}")
    suite = compute_var_suite(returns, window=window, alpha=alpha)
    if suite.empty:
        raise ValueError(f"Not enough data: returns={len(returns)} window={window}")
    suite = suite.iloc[-1].to_dict()
    feats = calculate_features(pd.DataFrame({'Close': close, 'returns': close.pct_change()}),
                               window=20)
    out = {
        'as_of': str(close.index[-1].date()),
        'window': window, 'alpha': alpha, 'confidence': 1 - alpha,
        'hist_var': float(suite['hist_var']),
        'param_var': float(suite['param_var']),
        'ewma_var': float(suite['ewma_var']),
        'es': float(suite['es']),
        'volatility_20d': float(feats['volatility'].iloc[-1]) if not feats.empty else None,
        'downside_vol_20d': float(feats['downside_vol'].iloc[-1]) if not feats.empty else None,
    }
    if gmm_path and os.path.exists(gmm_path):
        try:
            row = feats.iloc[-1]
            reg = predict_regime(row, path=gmm_path)
            out['regime'] = reg
        except Exception as e:
            out['regime_error'] = str(e)
    return out

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('SPY', period='3y', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    print(predict_var_latest(close, gmm_path=DEFAULT_GMM_PATH))
