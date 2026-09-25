import os
import argparse
import pickle
from data_loader import download_spy, load_spy_data, calculate_features, train_test_split
from var_calculator import calculate_all_var, regime_aware_var
from gmm_risk_states import predict_risk_states
from predict import fit_and_save_gmm
from anomaly_detector import detect_anomalies, merge_alerts
from portfolio_forecast import monte_carlo_paths, forecast_all_regimes
from backtester import run_all, backtest
import visualize as viz

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='SPY')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-11-27')
    p.add_argument('--window', type=int, default=500)
    p.add_argument('--feature-window', type=int, default=20)
    p.add_argument('--confidence', type=float, default=0.95)
    p.add_argument('--n-states', type=int, default=3)
    p.add_argument('--contamination', type=float, default=0.05)
    p.add_argument('--mc-horizon', type=int, default=20)
    p.add_argument('--mc-paths', type=int, default=10000)
    p.add_argument('--test-start', default='2023-01-01')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--skip-download', action='store_true')
    p.add_argument('--skip-plots', action='store_true')
    args = p.parse_args()

    alpha = 1.0 - args.confidence
    raw_dir = os.path.join(args.data_dir, 'raw')
    raw_path = os.path.join(raw_dir, f'{args.ticker.lower()}_raw.csv')
    fig_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.skip_download and os.path.exists(raw_path):
        print(f"Reusing: {raw_path}")
    else:
        download_spy(args.ticker, args.start, args.end, raw_dir)
    df = load_spy_data(raw_path)
    feats = calculate_features(df, window=args.feature_window)
    train, test = train_test_split(feats, args.test_start)

    var_df = calculate_all_var(feats['returns'], window=args.window, alpha=alpha)
    var_df.to_csv(os.path.join(args.out_dir, 'var_results.csv'))

    gmm_path = os.path.join('models', 'gmm_bundle.pkl')
    states, bundle = fit_and_save_gmm(feats, n_states=args.n_states, save_path=gmm_path, fit_end=args.test_start)
    gmm, scaler, names = bundle['gmm'], bundle['scaler'], bundle['names']
    states.to_csv(os.path.join(args.out_dir, 'risk_states.csv'))

    anom, iso = detect_anomalies(feats, contamination=args.contamination)
    merged = merge_alerts(states, anom)
    merged.to_csv(os.path.join(args.out_dir, 'alerts.csv'))

    test_var = var_df.loc[var_df.index.intersection(test.index)]
    aligned_states = states['risk_state'].reindex(feats.index).ffill()
    rav = regime_aware_var(feats['returns'], aligned_states, alpha=alpha)
    test_var = test_var.assign(regime_var=rav.loc[test_var.index])
    bt = run_all(test['returns'], test_var, alpha=alpha,
                 methods=('hist_var', 'param_var', 'ewma_var', 'regime_var'))
    bt.to_csv(os.path.join(args.out_dir, 'backtest_summary.csv'), index=False)

    aligned_returns = feats['returns'].loc[states.index]
    mc = forecast_all_regimes(aligned_returns, states['state_name'],
                              horizon=args.mc_horizon, n_paths=args.mc_paths, alpha=alpha)
    cur_regime = states['state_name'].iloc[-1]
    with open(os.path.join(args.out_dir, 'mc_summary.pkl'), 'wb') as f:
        pickle.dump({'per_regime': {k: {kk: vv for kk, vv in v.items() if kk != 'paths'}
                                     for k, v in mc.items()},
                     'current_regime': cur_regime}, f)

    if not args.skip_plots:
        os.makedirs(fig_dir, exist_ok=True)
        viz.plot_var_overlay(var_df, os.path.join(fig_dir, 'var_overlay.png'), alpha)
        viz.plot_violations(test['returns'], test_var['hist_var'],
                            os.path.join(fig_dir, 'violations_hist.png'), 'Historical')
        viz.plot_violations(test['returns'], test_var['ewma_var'],
                            os.path.join(fig_dir, 'violations_ewma.png'), 'EWMA')
        viz.plot_risk_states(feats, states, os.path.join(fig_dir, 'risk_states.png'))
        viz.plot_anomalies(feats, anom, os.path.join(fig_dir, 'anomalies.png'))
        viz.plot_mc_paths(mc, os.path.join(fig_dir, 'mc_paths.png'))
        viz.plot_backtest_summary(bt, os.path.join(fig_dir, 'backtest_summary.png'))

if __name__ == "__main__":
    main()