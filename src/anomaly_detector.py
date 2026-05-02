import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

IF_FEATS = ['returns', 'volatility', 'downside_vol', 'skew', 'kurt']

def detect_anomalies(features, contamination=0.05, feats=IF_FEATS, seed=42):
    cols = [c for c in feats if c in features.columns]
    valid = features[cols].dropna()
    Xs = StandardScaler().fit_transform(valid.values)
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    iso.fit(Xs)
    pred = iso.predict(Xs)            # 1 normal, -1 anomaly
    score = iso.score_samples(Xs)     # higher = more normal
    out = pd.DataFrame(index=valid.index)
    out['anomaly'] = (pred == -1).astype(int)
    out['anom_score'] = -score        # flip so higher = more anomalous
    n_anom = int(out['anomaly'].sum())
    print(f"IsolationForest: anomalies={n_anom}/{len(out)} ({n_anom/len(out):.2%})")
    return out, iso

def merge_alerts(states_df, anom_df):
    # combined alert: GMM stress regime OR IF anomaly
    df = states_df.join(anom_df, how='inner')
    stress_names = {n for n in df['state_name'].unique() if 'Stress' in n or 'Elevated' in n}
    df['gmm_stress'] = df['state_name'].isin(stress_names).astype(int)
    df['alert'] = ((df['gmm_stress'] == 1) | (df['anomaly'] == 1)).astype(int)
    return df

if __name__ == "__main__":
    from data_loader import load_spy_data, calculate_features
    df = load_spy_data('data/raw/spy_raw.csv')
    feats = calculate_features(df)
    anom, _ = detect_anomalies(feats)
    print(anom.tail())
