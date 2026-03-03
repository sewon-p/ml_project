import json
import numpy as np
import torch
import pandas as pd
from src.evaluation.metrics import compute_all_metrics
from src.models.cnn1d import CNN1DEstimator
from src.models.lstm import LSTMEstimator
from src.models.tabular import XGBoostEstimator

# 1. Load Data
data = np.load('data/features/timeseries.npz')
seq, target = data['sequences'], data['density']
scenario_ids = data['scenario_ids']

rng = np.random.RandomState(42)
unique_ids = np.unique(scenario_ids)
rng.shuffle(unique_ids)
n_test = max(1, int(len(unique_ids) * 0.2))
test_ids = set(unique_ids[:n_test])
test_idx = np.where(np.isin(scenario_ids, list(test_ids)))[0]

batch_size = 512
device = 'cuda'

X_test_np = seq[test_idx]
y_test = target[test_idx]

def predict_in_batches(model, X_np):
    all_preds = []
    for i in range(0, len(X_np), batch_size):
        X_batch = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
        preds = model(X_batch).cpu().numpy().flatten()
        all_preds.append(preds)
    return np.concatenate(all_preds)

metrics_dict = {}

# CNN1D
cnn = CNN1DEstimator.load('outputs/cnn1d/best_model.pt', input_size=6, hidden_channels=[32, 64], kernel_sizes=[3, 3])
cnn.model.to(device)
cnn.model.eval()
with torch.no_grad():
    y_pred_cnn = predict_in_batches(cnn.model, X_test_np)
metrics_dict["CNN1D"] = compute_all_metrics(y_test, y_pred_cnn)

# LSTM
lstm = LSTMEstimator.load('outputs_lstm/lstm_best.pt', input_size=6, hidden_size=64, num_layers=2)
lstm.model.to(device)
lstm.model.eval()
with torch.no_grad():
    y_pred_lstm = predict_in_batches(lstm.model, X_test_np)
metrics_dict["LSTM"] = compute_all_metrics(y_test, y_pred_lstm)

# XGBoost
df = pd.read_parquet('data/features/dataset.parquet')
test_df = df[df['scenario_id'].isin(test_ids)]
exclude = {"scenario_id", "probe_idx", "density", "flow", "demand_vehph", "k_fd", "q_fd", "delta_density", "delta_flow"}
f_cols = [c for c in df.columns if c not in exclude]

xgb = XGBoostEstimator.load('outputs_xgboost/xgboost_best.pkl')
y_pred_xgb = xgb.predict(test_df[f_cols].values)
metrics_dict["XGBoost"] = compute_all_metrics(test_df['density'].values, y_pred_xgb)

with open('outputs/metrics_summary.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)
print("Saved to outputs/metrics_summary.json")
