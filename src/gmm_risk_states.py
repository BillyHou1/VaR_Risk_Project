# GMM Risk State Identification

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# Fit GMM to features
def fit_gmm(features, n_states=3):
    feature_cols = ['volatility', 'downside_vol', 'returns']
    X = features[feature_cols].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=n_states, covariance_type='full',
                        n_init=10, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    print(f"GMM fitted, silhouette: {silhouette_score(X_scaled, labels):.4f}")
    return gmm, scaler, labels
# Assign state names based on volatility
def assign_state_names(X, labels, n_states=3):
    # rank states by mean volatility
    state_vol = {}
    for s in range(n_states):
        mask = labels == s
        state_vol[s] = X[mask, 0].mean()  # volatility is first col
    sorted_states = sorted(state_vol.keys(), key=lambda s: state_vol[s])
    names = {sorted_states[0]: 'Low Vol', sorted_states[1]: 'Normal', sorted_states[2]: 'High Vol'}
    return names
# Predict risk states for all dates
def predict_risk_states(features, n_states=3):
    feature_cols = ['volatility', 'downside_vol', 'returns']
    valid_idx = features[feature_cols].dropna().index
    X = features.loc[valid_idx, feature_cols].values
    gmm, scaler, labels = fit_gmm(features, n_states)
    names = assign_state_names(scaler.transform(X), labels, n_states)
    results = pd.DataFrame(index=valid_idx)
    results['risk_state'] = labels
    results['state_name'] = [names[s] for s in labels]
    probs = gmm.predict_proba(scaler.transform(X))
    for i in range(n_states):
        results[f'prob_state_{i}'] = probs[:, i]
    return results
# Example usage
if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    df = load_spy_data()
    features = calculate_features(df)
    states = predict_risk_states(features)
    print("\nRisk State Distribution:")
    print(states['state_name'].value_counts())
