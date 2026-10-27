"""Strict probability bridge for CAS cross-family experiments.

All outputs use the same nine quantiles on [0.1, 0.9]. Point-only
models are deliberately unsupported; quantile nodes are never treated as draws.
"""
import numpy as np
import torch

GRID = np.arange(1, 10, dtype=np.float64) / 10


def summarize_distribution(values, *, levels=None, samples=False):
    """Input [B,C,H,Q or S]; return P50 and integrated absolute deviation."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError("Expected finite [batch,channel,horizon,distribution] output")
    if samples:
        if values.shape[-1] < 100:
            raise ValueError("CAS sampling models require at least 100 genuine draws")
        quantiles = np.moveaxis(np.quantile(values, GRID, axis=-1), 0, -1)
        crossings = 0
    else:
        levels = np.asarray(levels, dtype=np.float64)
        if (levels.ndim != 1 or len(levels) != values.shape[-1] or len(levels) < 2
                or not np.isfinite(levels).all() or (np.diff(levels) <= 0).any()
                or levels[0] > GRID[0]+1e-7 or levels[-1] < GRID[-1]-1e-7):
            raise ValueError("Native quantiles must cover [0.1,0.9]; no tail extrapolation")
        # Predeclared monotone rearrangement, identical for all quantile families.
        crossings = int((np.diff(values, axis=-1) < 0).sum())
        values = np.sort(values, axis=-1)
        flat = values.reshape(-1, values.shape[-1])
        quantiles = np.stack([np.interp(GRID, levels, row) for row in flat])
        quantiles = quantiles.reshape(*values.shape[:-1], len(GRID))
    point = quantiles[..., 4]
    deviation = np.abs(quantiles-point[..., None])
    dispersion = ((deviation[..., :-1]+deviation[..., 1:])*.5*np.diff(GRID)).sum(-1)/.8
    return dict(forecast=torch.from_numpy(quantiles.astype(np.float32)),
                quantile_levels=GRID.tolist(), point_forecast=torch.from_numpy(point.astype(np.float32)),
                mean=torch.from_numpy(point.astype(np.float32)),
                uncertainty=torch.from_numpy(dispersion.astype(np.float32)),
                metadata=dict(distribution_protocol="cas_common_q10_q90_v1",
                              quantile_crossings_rearranged=crossings, genuine_samples=bool(samples)))


class CASDistributionAdapter:
    def __init__(self, family, model_path=None, device=None):
        if family == "chronos2":
            from .adapters.chronos2_adapter import Chronos2Adapter
            cls = Chronos2Adapter
        elif family == "moirai2":
            from .adapters.moirai2_adapter import Moirai2Adapter
            cls = Moirai2Adapter
        elif family == "sundial":
            from .adapters.sundial_adapter import SundialAdapter
            cls = SundialAdapter
        else:
            raise ValueError(f"Unsupported CAS probability family: {family}")
        self.family = family
        self.adapter = cls(model_path=model_path, device=device)

    def predict(self, context, forecast_horizon, num_samples=100, **kwargs):
        self.adapter._ensure_model_loaded()
        for name in ("model", "module"):
            component = getattr(self.adapter, name, None)
            if isinstance(component, torch.nn.Module):
                component.eval()
                component.requires_grad_(False)
        with torch.no_grad():
            result = self.adapter.predict(context=context, forecast_horizon=forecast_horizon,
                                          num_samples=num_samples, **kwargs)
        values = result["forecast"].detach().cpu().numpy()
        expected = (*context.shape[:2], forecast_horizon)
        if values.shape[:3] != expected:
            raise ValueError(f"{self.family} forecast shape {values.shape}, expected {expected} + distribution axis")
        if self.family == "sundial":
            return summarize_distribution(values, samples=True)
        levels = (result.get("quantile_levels") if self.family == "chronos2"
                  else getattr(self.adapter.module, "quantile_levels", None))
        if levels is None:
            raise ValueError(f"{self.family}: missing native quantile level metadata")
        return summarize_distribution(values, levels=levels)
