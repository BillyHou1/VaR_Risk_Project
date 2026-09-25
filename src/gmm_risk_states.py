import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

GMM_FEATS = ['volatility', 'downside_vol', 'returns']

def fit_gmm(features, n_states=3, feats=GMM_FEATS, seed=42):
    X = features[feats].dropna().values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=n_states, covariance_type='full', n_init=10, random_state=seed)
    gmm.fit(Xs)
    labels = gmm.predict(Xs)
    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else float('nan')
    print(f"GMM fit: states={n_states}, silhouette={sil:.4f}")
    return gmm, scaler, labels, Xs

def assign_state_names(Xs, labels, n_states):
    # rank states by mean volatility (z-scored col 0)
    order = sorted(range(n_states), key=lambda s: Xs[labels == s, 0].mean())
    bank = {2: ['Calm', 'Stress'],
            3: ['Calm', 'Normal', 'Stress'],
            4: ['Calm', 'Normal', 'Elevated', 'Stress'],
            5: ['Calm', 'Quiet', 'Normal', 'Elevated', 'Stress']}.get(n_states,
                  [f'S{i}' for i in range(n_states)])
    return {s: bank[i] for i, s in enumerate(order)}

def predict_risk_states(features, n_states=3, fit_end=None):
    valid_idx = features[GMM_FEATS].dropna().index
    fit_idx = valid_idx[valid_idx < pd.Timestamp(fit_end)] if fit_end else valid_idx
    gmm, scaler, fit_labels, fit_Xs = fit_gmm(features.loc[fit_idx], n_states)
    names = assign_state_names(fit_Xs, fit_labels, n_states)
    Xs = scaler.transform(features.loc[valid_idx, GMM_FEATS].values)
    labels = gmm.predict(Xs)
    df = pd.DataFrame(index=valid_idx)
    df['risk_state'] = labels
    df['state_name'] = [names[s] for s in labels]
    probs = gmm.predict_proba(Xs)
    for i in range(n_states):
        df[f'prob_state_{i}'] = probs[:, i]
    return df, gmm, scaler, names

if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    df = load_spy_data('data/raw/spy_raw.csv')
    feats = calculate_features(df)
    states, *_ = predict_risk_states(feats)
    print(states['state_name'].value_counts())