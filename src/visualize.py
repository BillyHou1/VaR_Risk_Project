import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = ['#2E7D32', '#1976D2', '#FBC02D', '#E64A19', '#6A1B9A']

def plot_var_overlay(var_df, out_path, alpha=0.05):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(var_df.index, var_df['returns'], color='black', lw=0.5, alpha=0.55, label='returns')
    for c, col in zip(['hist_var', 'param_var', 'ewma_var'], ['#1976D2', '#E64A19', '#2E7D32']):
        if c in var_df.columns:
            ax.plot(var_df.index, var_df[c], lw=1.0, label=c, color=col)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_title(f'Daily returns vs VaR (alpha={alpha})')
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_violations(returns, var, out_path, label='Historical'):
    idx = returns.index.intersection(var.index)
    r, v = returns.loc[idx], var.loc[idx]
    viol = r < v
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(idx, r, color='black', lw=0.5, alpha=0.6, label='returns')
    ax.plot(idx, v, color='#E64A19', lw=1.0, label=f'{label} VaR')
    ax.scatter(idx[viol], r[viol], color='red', s=14, zorder=3, label=f'violations (n={int(viol.sum())})')
    ax.set_title(f'{label} VaR violations'); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_risk_states(features, states, out_path):
    df = features.join(states[['state_name']], how='inner')
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, df['close'], color='black', lw=0.7)
    names = sorted(df['state_name'].unique())
    cmap = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    lo, hi = df['close'].min(), df['close'].max()
    for n in names:
        m = df['state_name'] == n
        ax.fill_between(df.index, lo, hi, where=m, color=cmap[n], alpha=0.18, label=n)
    ax.set_title('SPY price with GMM risk regimes')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_anomalies(features, anom_df, out_path):
    df = features.join(anom_df, how='inner')
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df.index, df['close'], color='black', lw=0.6)
    flags = df[df['anomaly'] == 1]
    ax.scatter(flags.index, flags['close'], color='red', s=10, alpha=0.8, label=f'anomaly (n={len(flags)})')
    ax.set_title('IsolationForest anomalies on price')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_mc_paths(mc_results, out_path, n_show=200):
    fig, axes = plt.subplots(1, len(mc_results), figsize=(5.2 * len(mc_results), 4.2), sharey=True)
    if len(mc_results) == 1: axes = [axes]
    for ax, (name, res) in zip(axes, mc_results.items()):
        paths = res['paths'][:n_show]
        for p in paths: ax.plot(p, color='gray', lw=0.4, alpha=0.4)
        ax.plot(paths.mean(axis=0), color='red', lw=1.2, label='mean')
        ax.set_title(f"{name}\nVaR={res['var_h']:.3f}  ES={res['es_h']:.3f}")
        ax.set_xlabel('day'); ax.legend(fontsize=8)
    axes[0].set_ylabel('cumulative growth')
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_backtest_summary(bt_df, out_path):
    if bt_df.empty: return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bt_df))
    ax.bar(x - 0.2, bt_df['viol_rate'], 0.4, label='violation rate', color='#1976D2')
    ax.axhline(0.05, color='red', lw=1.0, ls='--', label='alpha=5%')
    ax.set_xticks(x); ax.set_xticklabels(bt_df['label'], rotation=15)
    ax.set_ylabel('violation rate'); ax.set_title('Backtest summary'); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")