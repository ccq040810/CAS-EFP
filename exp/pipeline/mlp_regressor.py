"""MLP backend: same directional objective as regressor.py, trained by autograd."""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn


def directional_loss(pred, target, weight, *, alpha=2.0, direction=1, asymmetric=True):
    error = pred - target
    if asymmetric:
        coefficient = torch.where(-error * direction > 0, alpha * weight, torch.ones_like(error))
    else:
        coefficient = torch.ones_like(error)
    return (0.5 * coefficient * error.square()).mean()


class MLPRegressor:
    def __init__(self, *, hidden=(128, 64), lr=1e-3, batch_size=8192,
                 epochs=300, patience=30, seed=42, device="cpu", threads=4,
                 alpha=2.0, direction=1, asymmetric=False, selection="legacy"):
        if direction not in (-1, 1) or alpha < 1 or not np.isfinite(alpha):
            raise ValueError("direction must be +/-1 and alpha must be finite >=1")
        if selection not in ("legacy", "rmse"):
            raise ValueError("unknown selection metric")
        if not hidden or min(hidden) < 1 or min(batch_size, epochs, patience, threads) < 1 or lr <= 0:
            raise ValueError("invalid MLP hyperparameters")
        self.config = dict(hidden=list(hidden), lr=lr, batch_size=batch_size,
                           epochs=epochs, patience=patience, seed=seed, device=device,
                           threads=threads, alpha=alpha, direction=direction,
                           asymmetric=asymmetric, selection=selection)
        self.device = torch.device(device)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, self.fill)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit(self, X, y, X_valid, y_valid, sample_weight=None):
        c = self.config
        random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
        torch.set_num_threads(c["threads"])
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use --gpus cpu for CPU")
            torch.cuda.manual_seed_all(c["seed"])
        torch.use_deterministic_algorithms(True)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1)
        if not len(y) or not len(y_valid) or not np.isfinite(y).all() or not np.isfinite(y_valid).all():
            raise ValueError("nonempty, finite train and validation targets required")
        if len(X) != len(y) or len(X_valid) != len(y_valid):
            raise ValueError("feature/label length mismatch")
        # All preprocessing is fitted only on training data. Missing indicators
        # are intentionally NOT added: keep the original feature columns.
        self.fill = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0
                              for col in X.T], dtype=np.float32)
        clean = np.where(np.isfinite(X), X, self.fill)
        self.mean = clean.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = clean.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-8] = 1
        self.y_mean = float(y.mean(dtype=np.float64))
        self.y_std = max(float(y.std(dtype=np.float64)), 1e-8)
        widths = [X.shape[1], *c["hidden"], 1]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i+1]))
            if i < len(widths)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers).to(self.device)
        # Retain large datasets on CPU; only each minibatch occupies GPU memory.
        xt = torch.from_numpy(self.transform(X))
        yt = torch.from_numpy((y - self.y_mean) / self.y_std)
        w = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("invalid sample weights")
        wt = torch.from_numpy(w)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=c["lr"])
        generator = torch.Generator().manual_seed(c["seed"])
        best, stale, state = float("inf"), 0, None
        self.history = []
        for epoch in range(1, c["epochs"]+1):
            self.network.train()
            order = torch.randperm(len(y), generator=generator)
            total = 0.0
            for idx in order.split(c["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                pred = self.network(xt[idx].to(self.device)).squeeze(-1)
                loss = directional_loss(pred, yt[idx].to(self.device), wt[idx].to(self.device),
                                        alpha=c["alpha"], direction=c["direction"], asymmetric=c["asymmetric"])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(idx)
            prediction = self.predict(X_valid)
            error = prediction - y_valid
            # Match existing LGBM validation: error*alpha BEFORE squaring;
            # validation has no confidence weights. Optional RMSE is explicit.
            if c["selection"] == "legacy" and c["asymmetric"]:
                error = np.where(-error*c["direction"] > 0, error*c["alpha"], error)
            score = float(np.sqrt(np.mean(error.astype(np.float64)**2)))
            if not np.isfinite(score):
                raise RuntimeError("nonfinite validation score")
            self.history.append(dict(epoch=epoch, train_loss_normalized=total/len(y), valid_selection=score))
            if score < best:
                best, stale = score, 0
                state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                self.best_iteration = epoch
            else:
                stale += 1
            print(f"epoch={epoch} train_loss={total/len(y):.6g} valid={score:.6g} best={self.best_iteration}", flush=True)
            if stale >= c["patience"]:
                break
        self.network.load_state_dict(state)
        self.best_score = best
        return self

    def predict(self, X):
        self.network.eval()
        values = torch.from_numpy(self.transform(X))
        chunks = []
        with torch.no_grad():
            for batch in values.split(self.config["batch_size"]):
                chunks.append(self.network(batch.to(self.device)).squeeze(-1).cpu().numpy())
        return np.concatenate(chunks).astype(np.float64) * self.y_std + self.y_mean if chunks else np.empty(0)

    def save(self, path):
        torch.save(dict(config=self.config, state_dict={k:v.detach().cpu() for k,v in self.network.state_dict().items()},
                        fill=self.fill, mean=self.mean, std=self.std, y_mean=self.y_mean,
                        y_std=self.y_std, best_epoch=self.best_iteration), path)
