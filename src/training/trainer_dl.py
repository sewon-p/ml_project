"""PyTorch training loop for deep learning models (CNN1D, LSTM)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class DLTrainer:
    """Generic PyTorch training loop with early stopping and TensorBoard."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        scheduler_config: dict[str, Any] | None = None,
        max_epochs: int = 200,
        patience: int = 20,
        checkpoint_dir: str | Path | None = None,
        tensorboard_dir: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.history: list[dict[str, float]] = []

        # Optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
            )

        # Scheduler
        self.scheduler = None
        if scheduler_config:
            stype = scheduler_config.get("type", "cosine")
            if stype == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get("T_max", max_epochs),
                )
            elif stype == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get("step_size", 50),
                    gamma=scheduler_config.get("gamma", 0.1),
                )

        # TensorBoard
        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import (
                    SummaryWriter,
                )

                tb_path = Path(tensorboard_dir)
                tb_path.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(tb_path))
            except ImportError:
                logger.warning("TensorBoard not available, skipping.")

        self.criterion = nn.MSELoss()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Run the training loop."""
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            record: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
                record["val_loss"] = val_loss
                record.update({f"val_{k}": v for k, v in val_metrics.items()})

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    if self.checkpoint_dir:
                        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        save_checkpoint(
                            self.model.state_dict(),
                            self.checkpoint_dir / "best_model.pt",
                        )
                else:
                    self.epochs_no_improve += 1

            if self.scheduler:
                self.scheduler.step()

            self.history.append(record)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if "val_loss" in record:
                    self.writer.add_scalar(
                        "Loss/val",
                        record["val_loss"],
                        epoch,
                    )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f%s",
                    epoch,
                    self.max_epochs,
                    train_loss,
                    (f" val_loss={record.get('val_loss', 0):.4f}" if val_loader else ""),
                )

            if self.epochs_no_improve >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if self.writer:
            self.writer.close()

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

    @staticmethod
    def _unpack_batch(
        batch: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Unpack 2-tuple (X, y) or 3-tuple (X, y, cond) from DataLoader."""
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        nb = self.device.type != "cpu"  # non_blocking for GPU/MPS
        for batch in loader:
            X_batch, y_batch, cond = self._unpack_batch(batch)
            X_batch = X_batch.to(self.device, non_blocking=nb)
            y_batch = y_batch.to(self.device, non_blocking=nb)
            if cond is not None:
                cond = cond.to(self.device, non_blocking=nb)
            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(X_batch, cond)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        nb = self.device.type != "cpu"
        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch, cond = self._unpack_batch(batch)
                X_batch = X_batch.to(self.device, non_blocking=nb)
                y_batch = y_batch.to(self.device, non_blocking=nb)
                if cond is not None:
                    cond = cond.to(self.device, non_blocking=nb)
                preds = self.model(X_batch, cond)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()
                n_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        avg_loss = total_loss / max(n_batches, 1)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        metrics = compute_all_metrics(y_true, y_pred)
        return avg_loss, metrics

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Generate predictions for an entire DataLoader."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                X_batch, _, cond = self._unpack_batch(batch)
                X_batch = X_batch.to(self.device)
                if cond is not None:
                    cond = cond.to(self.device)
                preds = self.model(X_batch, cond)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)
