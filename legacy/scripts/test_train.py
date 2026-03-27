"""Quick test training script."""
import logging

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import create_model
from src.training.trainer_dl import DLTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

data = np.load("data/features_test/timeseries.npz")
sequences = data["sequences"]
targets = data["density"]

n = len(sequences)
train_n = int(n * 0.7)
val_n = int(n * 0.15)

train_ds = TimeSeriesDataset(sequences[:train_n], targets[:train_n])
val_ds = TimeSeriesDataset(sequences[train_n:train_n+val_n], targets[train_n:train_n+val_n])
test_ds = TimeSeriesDataset(sequences[train_n+val_n:], targets[train_n+val_n:])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

estimator = create_model("cnn1d", in_channels=6, seq_len=300, device="cpu")
model = estimator.model

trainer = DLTrainer(model=model, device="cpu", max_epochs=5, patience=3, learning_rate=0.001)
results = trainer.fit(train_loader, val_loader)
print(f"Best val_loss: {results['best_val_loss']:.4f}")

test_preds = trainer.predict(test_loader)
test_metrics = compute_all_metrics(targets[train_n+val_n:], test_preds)
print(f"Test metrics: {test_metrics}")
