# Main Pipeline

from data_loader import load_spy_data, calculate_features, train_test_split
from var_calculator import calculate_all_var
from gmm_risk_states import predict_risk_states
from backtester import backtest

print("Loading data...")
df = load_spy_data()
features = calculate_features(df)
train, test = train_test_split(features)

print("\nCalculating VaR...")
var_results = calculate_all_var(features['returns'])

print("\nIdentifying risk states...")
risk_states = predict_risk_states(features)
print(risk_states['state_name'].value_counts())

print("\nBacktesting on test set...")
test_var = var_results.loc[test.index]

print("\nHistorical VaR:")
backtest(test['returns'], test_var['hist_var'])

print("\nParametric VaR:")
backtest(test['returns'], test_var['param_var'])

print("\nEWMA VaR:")
backtest(test['returns'], test_var['ewma_var'])

print("\nDone!")
