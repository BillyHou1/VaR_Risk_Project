# Integrated Market Risk Monitor: VaR and Machine Learning

Boyu Hou, University of Bristol

Four one-day 95% Value-at-Risk (VaR) forecasts for SPY are compared in this project, using daily prices from January 2010 to November 2025. Historical simulation and a parametric normal model are estimated on a rolling 500-day window. The EWMA model follows RiskMetrics with a decay factor of 0.94. The fourth forecast is regime-aware. Specifically, a three-component Gaussian mixture model (GMM) is fitted on 20-day volatility, 20-day downside volatility and the daily return, and its states are named Calm, Normal and Stress by volatility. The mean and standard deviation of the parametric VaR are then taken only from past days in the same regime. All forecasts for day t use data up to day t-1, and the GMM is fitted on data before 2023 only. An Isolation Forest flag and a 20-day Monte Carlo simulation per regime are also produced. However, both are descriptive and are not part of the backtest.

## Results

The backtest is run over 729 trading days from January 2023 to November 2025, and each violation rate is compared with the nominal 5%. Violation rates of 5.49% and 5.76% are obtained for EWMA and the regime-aware forecast, and both are accepted by the Kupiec and Christoffersen tests at the 5% level. Historical simulation is found to be too conservative. Its violation rate is 3.43%, and it is rejected by the Kupiec test (p = 0.040). The parametric model gives 3.57% and is accepted only narrowly (p = 0.062). The regime-aware forecast is therefore acceptable, but it is not closer to the nominal rate than EWMA. Only one asset and one train-test split are used. The Stress state is first assigned on 27 February 2020, six trading days after the 19 February peak. However, this date lies inside the training period, so it is not an out-of-sample warning.

| Method | Violations (n = 729) | Rate | Kupiec p | Christoffersen p |
|---|---:|---:|---:|---:|
| Historical simulation | 25 | 3.43% | 0.040 | 0.271 |
| Parametric normal | 26 | 3.57% | 0.062 | 0.312 |
| EWMA | 40 | 5.49% | 0.552 | 0.585 |
| Regime-aware (GMM) | 42 | 5.76% | 0.357 | 0.767 |

## Usage

```bash
pip install -r requirements.txt
python src/main.py
```

## References

- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- RiskMetrics Group (1996). *RiskMetrics Technical Document*. J.P. Morgan.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation Forest. IEEE ICDM, pp. 413-422.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.