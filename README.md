# Integrated Market Risk Monitor: VaR & Machine Learning

## 1. Project Abstract

Value-at-Risk (VaR) is how banks and regulators measure market risk. The problem is traditional VaR reacts slowly when market conditions shift. During March 2020, the S&P 500 fell 34% over 23 trading days. Standard VaR estimates, calibrated on pre-crisis data, failed to reflect this shift in real time.

This project implements a hybrid approach. I integrate Gaussian Mixture Models (GMM) with parametric VaR to enable regime-aware risk estimation. The GMM component identifies latent market states (Calm, Normal, Stress) without supervised labels. The VaR calculation then adjusts based on the detected regime.

## 2. Methodology

**Dataset:** SPY (S&P 500 ETF) daily OHLCV data, January 2011 to November 2025. Approximately 3,500 observations.

### A. Baseline VaR Models

I implemented three conventional VaR methods in `src/var_calculator.py` as control benchmarks:

**Historical Simulation:** 500-day rolling window, non-parametric. Captures realized tail events but adapts slowly to new regimes.

**Parametric VaR:** Assumes returns ~ N(μ, σ). I retained this method despite its underestimation of fat-tailed events because it serves as the industry standard baseline.

**EWMA:** Decay factor λ = 0.94 per RiskMetrics (1996). Assigns greater weight to recent observations, improving volatility responsiveness.

### B. Machine Learning Components

**GMM Regime Detection** (`src/gmm_risk_states.py`)

I fit a 3-component GMM on volatility and volume features. The clustering is unsupervised. I never told it which days were "crisis" days.

Empirical observation: the GMM identified March 2020 as a Stress regime within 4 trading days of the drawdown onset. It also flagged the 2022 rate-hike driven correction. Neither event was explicitly labeled in training.

**Isolation Forest Anomaly Detection** (`src/anomaly_detector.py`)

Isolation Forest detects statistical outliers that fall outside normal cluster behavior. I use this as a secondary alert mechanism. Days flagged by Isolation Forest receive additional scrutiny regardless of GMM classification.

### C. Validation Framework

**Walk-Forward Backtesting** (`src/backtest.py`)

I train on historical data, test on future data, then roll the window forward. The model never sees tomorrow's prices during training.

**Monte Carlo Simulation** (`src/portfolio_forecast.py`)

10,000 path simulations conditioned on current regime state. If GMM indicates Stress, the simulation draws from stress-period volatility parameters rather than the unconditional 14-year average.

## 3. Usage

**Requirements:** Python 3.8+, scikit-learn, pandas, numpy, scipy

```bash
pip install -r requirements.txt
python src/main.py --ticker SPY --window 500 --confidence 0.95
```

## 4. References

- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- RiskMetrics Group (1996). *RiskMetrics Technical Document*. J.P. Morgan.
- Liu, F., Ting, K., & Zhou, Z. (2008). "Isolation Forest". IEEE ICDM, pp. 413–422.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.

---

*Billy Hou | University of Bristol | 2025*
