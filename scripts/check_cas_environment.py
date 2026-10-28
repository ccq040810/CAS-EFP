"""Offline preflight: inspect all three upstreams and both downstream backends.

Never installs packages, downloads weights, prepares features or starts experiments.
Each model is probed in a separate process, optionally with its own interpreter.
Only report/log files are written. Exit 0 means every required check passed; 2
means at least one item is missing, failed, or not verified by real inference.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = {
    "chronos2": ("amazon/chronos-2", "chronos.chronos2"),
    "moirai2": ("Salesforce/moirai-2.0-R-small", "uni2ts.model.moirai2"),
    "sundial": ("thuml/sundial-base-128m", "transformers"),
}
COMMON = ["numpy", "pandas", "pyarrow", "lightgbm", "sklearn", "chinese_calendar", "torch"]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def inspect_weights(folder):
    folder = Path(folder)
    errors = []
    config = folder/"config.json"
    if not config.is_file():
        errors.append(f"Missing config.json: {folder}")
    weights = [p for p in folder.glob("*") if p.suffix in (".safetensors", ".bin", ".pt", ".pth")]
    if not weights:
        errors.append(f"No local weight files: {folder}")
    for path in weights:
        if path.stat().st_size == 0:
            errors.append(f"Empty weight file: {path}")
    for index in folder.glob("*.index.json"):
        try:
            mapping = json.loads(index.read_text()).get("weight_map", {})
            for filename in set(mapping.values()):
                if not (folder/filename).is_file():
                    errors.append(f"Missing weight shard: {folder/filename}")
        except Exception as exc:
            errors.append(f"Unreadable shard index {index}: {exc}")
    return dict(path=str(folder), weight_files=len(weights), errors=errors)


def inspect_inputs(old_root):
    errors, rows = [], []
    for market, name in (("guangdong", "GD/guangdong_aligned.csv"),
                         ("reale_delu", "REALE/DE-LU_aligned.csv")):
        old, new = Path(old_root)/"data"/name, ROOT/"data"/name
        same = old.is_file() and new.is_file() and os.path.samefile(old, new)
        if not same:
            errors.append(f"Data path must resolve to original file: {new} -> {old}")
        rows.append(dict(market=market, original=str(old), current=str(new), same_file=same))
        for window in (1, 2, 3):
            path = Path(old_root)/"results/rolling_adaptive_v3"/market/f"window_{window}/data_split.json"
            try:
                split = json.loads(path.read_text())[0]["data_split"]
                import datetime
                values = [datetime.datetime.fromisoformat(split[k]) for k in
                          ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")]
                if not (values[0] < values[1] <= values[2] < values[3] <= values[4] < values[5]):
                    raise ValueError("Overlapping or unordered train/validation/test boundaries")
                rows.append(dict(market=market, window=window, split=str(path), status="OK"))
            except Exception as exc:
                errors.append(f"Invalid split {path}: {exc}")
    return dict(ready=not errors, rows=rows, errors=errors)


def probe(args):
    result = dict(component=args.probe, python=sys.executable, imports={}, errors=[], warnings=[], ready=False)
    try:
        modules = COMMON + ([] if args.probe == "downstream" else [FAMILIES[args.probe][1]])
        for name in modules:
            try:
                module = importlib.import_module(name)
                result["imports"][name] = dict(status="OK", version=getattr(module, "__version__", "unknown"))
            except Exception as exc:
                result["imports"][name] = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                result["errors"].append(f"Import {name}: {type(exc).__name__}: {exc}")
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["visible_gpus"] = torch.cuda.device_count()
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.expected_gpus:
            result["errors"].append(f"Expected {args.expected_gpus} visible CUDA GPUs, got {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            value = torch.ones(4, device=args.device)*2
            assert value.sum().item() == 8
            result["cuda_smoke"] = "OK"
            result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if args.probe == "downstream":
            if not result["errors"]:
                import numpy as np
                from exp.pipeline.mlp_regressor import MLPRegressor
                from exp.pipeline.cas_lgbm_regressor import CASLGBMRegressor
                rng = np.random.default_rng(42)
                X = rng.normal(size=(128, 3)).astype(np.float32)
                y = 2*X[:, 0]+X[:, 1]
                weights = np.linspace(1, 2, 96).astype(np.float32)
                mlp = MLPRegressor(hidden=(8,), epochs=2, patience=1, batch_size=32,
                                   threads=1, device=args.device, asymmetric=True, direction=-1, selection="rmse")
                tree = CASLGBMRegressor(seed=42, threads=1, alpha=2, direction=-1,
                                        asymmetric=True, selection="rmse", rounds=3, patience=2)
                for name, model in (("mlp", mlp), ("lightgbm", tree)):
                    model.fit(X[:96],y[:96],X[96:],y[96:],weights)
                    assert np.isfinite(model.predict(X[96:])).all()
                    result[name + "_real_training"] = "OK"
                # A synthetic model load alone cannot detect timestamp resolution
                # mismatches. Exercise the actual history selector in ns and us.
                import pandas as pd
                from exp.pipeline.tsfm_infer import _predict_multivar
                class HistoryProbe:
                    def predict(self,context,forecast_horizon,**kwargs):
                        np.testing.assert_array_equal(context.cpu().numpy(),[[[4.,5.,5.,7.]]])
                        values = context[..., -1:].expand(-1,-1,forecast_horizon)
                        return dict(mean=values,uncertainty=torch.ones_like(values))
                for unit in ("ns","us"):
                    index = pd.date_range("2024-01-01",periods=12,freq="h",tz="Asia/Shanghai").as_unit(unit)
                    frame = pd.DataFrame({"y":np.arange(12,dtype=float)},index=index)
                    frame.iloc[6,0] = np.nan
                    _predict_multivar(df=frame,mv_cols=["y"],anchors=[index[8]],anchor_offset={index[8]:0},
                                      model=HistoryProbe(),seq_len=4,horizon=2,batch_size=1,num_samples=1,
                                      device="cpu",use_future_covariates=False,causal_context_fill=True)
                    assert np.isnan(frame.iloc[6,0])
                result["rollout_origin_causality_ns_us"] = "OK"
        else:
            from huggingface_hub import snapshot_download
            requested = args.model_path or FAMILIES[args.probe][0]
            local = Path(requested).expanduser()
            if not local.is_dir():
                try:
                    local = Path(snapshot_download(requested, local_files_only=True))
                except Exception as exc:
                    result["errors"].append(f"Weights not found locally or in HF cache: {requested}: {exc}")
                    local = None
            if local is not None:
                result["weights"] = inspect_weights(local)
                result["errors"].extend(result["weights"]["errors"])
            if args.probe == "sundial":
                version = importlib.metadata.version("transformers")
                if version != "4.40.1":
                    result["warnings"].append(f"Sundial official quickstart specifies transformers==4.40.1; current {version}. Real inference is required; do not downgrade the Chronos environment in place.")
            if args.skip_inference:
                result["errors"].append("Real inference skipped; environment is not certified ready")
            elif not result["errors"]:
                from ts_models.cas_distribution import CASDistributionAdapter
                torch.manual_seed(42)
                context = torch.sin(torch.arange(168, device=args.device).float()/12).reshape(1,1,168)
                model = CASDistributionAdapter(args.probe, model_path=str(local), device=args.device)
                output = model.predict(context=context, forecast_horizon=24, num_samples=100)
                for key in ("point_forecast", "uncertainty"):
                    tensor = output[key]
                    assert tuple(tensor.shape) == (1,1,24), (key, tensor.shape)
                    assert torch.isfinite(tensor).all(), key
                assert (output["uncertainty"] >= 0).all()
                result["real_inference"] = "OK"
                result["distribution_protocol"] = output["metadata"]["distribution_protocol"]
        result["ready"] = not result["errors"]
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    save(args.report, result)
    return 0 if result["ready"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=ROOT.parent/"CAS-EFP")
    parser.add_argument("--output-root", type=Path, default=ROOT/"results/environment_check")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip-inference", action="store_true")
    for name in FAMILIES:
        parser.add_argument(f"--{name}-python", default=sys.executable)
        parser.add_argument(f"--{name}-path")
    parser.add_argument("--probe", choices=["downstream", *FAMILIES])
    parser.add_argument("--model-path")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.probe:
        return probe(args)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = dict(inputs=inspect_inputs(args.old_root), components={}, all_ready=False)
    for name in ("downstream", *FAMILIES):
        python = sys.executable if name == "downstream" else getattr(args, name + "_python")
        dest = args.output_root/(name + ".json")
        cmd = [python, str(Path(__file__).resolve()), "--probe", name, "--report", str(dest),
               "--device", args.device, "--expected-gpus", str(args.expected_gpus)]
        if name != "downstream":
            path = getattr(args, name + "_path")
            candidate = args.old_root/"models"/{"chronos2":"chronos-2", "moirai2":"moirai-2.0-R-small", "sundial":"sundial-base-128m"}[name]
            path = path or (str(candidate) if candidate.is_dir() else FAMILIES[name][0])
            cmd += ["--model-path", path]
        if args.skip_inference:
            cmd.append("--skip-inference")
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        print(f"CHECK {name} using {python}", flush=True)
        try:
            with (args.output_root/(name + ".log")).open("w", encoding="utf-8") as log:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout)
            # A stale report must never certify a crashed check as ready.
            if completed.returncode not in (0, 2) or not dest.is_file():
                raise RuntimeError(f"Probe exited {completed.returncode}; inspect {name}.log")
            item = json.loads(dest.read_text(encoding="utf-8"))
            if completed.returncode != 0:
                item["ready"] = False
        except Exception as exc:
            item = dict(ready=False, errors=[str(exc)])
        report["components"][name] = item
        print("READY" if item["ready"] else "NOT READY", name, flush=True)
        for error in item.get("errors", []):
            print("  " + error, flush=True)
        for warning in item.get("warnings", []):
            print("  NOTE: " + warning, flush=True)
    report["all_ready"] = report["inputs"]["ready"] and all(x["ready"] for x in report["components"].values())
    save(args.output_root/"report.json", report)
    lines = ["CAS environment preflight", "", f"ALL READY: {report['all_ready']}",
             f"Original data and six chronological splits: {report['inputs']['ready']}"]
    lines.extend(report["inputs"]["errors"])
    for name, item in report["components"].items():
        lines += ["", f"{name}: {'READY' if item['ready'] else 'NOT READY'}"]
        lines.extend("  " + e for e in item.get("errors", []))
        lines.extend("  NOTE: " + e for e in item.get("warnings", []))
    (args.output_root/"summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"Reports: {args.output_root.resolve()}", flush=True)
    return 0 if report["all_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
