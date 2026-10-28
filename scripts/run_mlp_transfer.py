"""Prepare identical CAS features once, then run independent MLP jobs per GPU.

No changes to the original LGBM pipeline. See docs/MLP_TRANSFER.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = {
    "tsfm_point": (False, False, False),
    "tsfm_point_C": (True, False, False),
    "tsfm_point_fixed_asym_noC": (False, True, False),
    "tsfm_point_C_fixed_asym": (True, True, False),
    "tsfm_point_C_confidence_asym": (True, True, True),
}
CORE = ["tsfm_point_C", "tsfm_point_C_fixed_asym", "tsfm_point_C_confidence_asym"]
ORIGINAL = list(METHODS)
ROLLOUT_PROTOCOL = "timestamp_searchsorted_causal_history_ffill_v2"
METHODS.update({
    "tsfm_point_C_mean_weight_asym": (True, True, True),
    "tsfm_point_C_shuffled_weight_asym": (True, True, True),
})


def suite_methods(suite):
    return CORE if suite == "core" else ORIGINAL if suite == "original" else list(METHODS)


def training_weights(confidence, method, weight_lambda, seed):
    """Controls change loss weights only; original confidence features stay intact."""
    import numpy as np
    confidence = np.asarray(confidence, dtype=np.float64)
    if (not np.isfinite(confidence).all() or (confidence < 0).any()
            or (confidence > 1).any()):
        raise ValueError("Confidence must be finite and within [0, 1]")
    if not np.isfinite(weight_lambda) or weight_lambda < 0:
        raise ValueError("weight_lambda must be finite and nonnegative")
    if not METHODS[method][2]:
        return None
    weights = 1 + weight_lambda * (1 - confidence)
    if method == "tsfm_point_C_mean_weight_asym":
        weights = np.full_like(weights, weights.mean())
    elif method == "tsfm_point_C_shuffled_weight_asym":
        weights = np.random.default_rng(seed).permutation(weights)
    return weights.astype(np.float32)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_config(args, manifest, method, seed):
    include_c = METHODS[method][0]
    names = manifest["feature_names"]
    return dict(method=method, seed=seed, alpha=args.alpha, weight_lambda=args.weight_lambda,
                regressor=getattr(args, "regressor", "mlp"),
                lgbm_rounds=getattr(args, "lgbm_rounds", 2000),
                lgbm_patience=getattr(args, "lgbm_patience", 100),
                upstream=manifest.get("upstream", "chronos2"),
                hidden=args.hidden, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                patience=args.patience, selection=args.selection, direction=manifest["direction"],
                feature_names=[n for i,n in enumerate(names) if include_c or i != manifest["confidence_index"]],
                prepared_sha256=manifest["arrays_sha256"])


def prepare(args):
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from scripts.run_rolling_adaptive_experiment import DATASETS
    from exp.pipeline.tsfm_infer import run_tsfm_rollout, _get_cache_path
    from exp.pipeline.feature_select import build_features
    np.random.seed(42)
    torch.manual_seed(42)
    upstream = getattr(args, "upstream", "chronos2")
    for market in args.markets:
        data, _, regress, tsfm, _ = DATASETS[market]
        for window in args.windows:
            folder = args.prepared_root / market / f"window_{window}"
            if folder.exists():
                raise FileExistsError(f"Prepared directory already exists: {folder}. Reuse it with run; prepare into a new directory to rebuild.")
            split = args.split_root / market / f"window_{window}" / "data_split.json"
            if not split.is_file():
                raise FileNotFoundError(f"Exact original split required: {split}")
            folder.mkdir(parents=True)
            write(folder / "data_split.json", read(split))
            pipeline = SimpleNamespace(
                data_path=str(ROOT/data), regress_cols_path=str(ROOT/regress), tsfm_cols_path=str(ROOT/tsfm),
                data_split_path=str(folder/"data_split.json"), is_std=True, scale=False,
                seq_len=168, pred_len=24, batch_size=args.tsfm_batch_size, tsfm_num_samples=100,
                enable_tsfm=True, tsfm_models=upstream, tsfm_model_paths=json.dumps({upstream: args.model_path}),
                tsfm_cache_path=str(args.cache_root/market/f"window_{window}" if upstream == "chronos2"
                                    else args.cache_root/upstream/market/f"window_{window}"),
                tag=f"{market}_rolling_w{window}", device=args.device, tsfm_use_future_covariates=0,
                tsfm_force_current_rerun=False, use_tsfm_point=True, use_tsfm_uncertainty=False,
                use_tsfm_confidence=True, use_ar_features=True, ar_lags="1,24,168", ar_roll="24",
                std_merge_valid_to_train=False, save_dir=str(folder), target_hour=0,
            )
            if upstream.startswith("cas_"):
                cache_dir = Path(pipeline.tsfm_cache_path)
                cache_dir.mkdir(parents=True, exist_ok=True)
                identity = dict(upstream=upstream, model_path=args.model_path,
                                protocol="cas_common_q10_q90_v1", seed=42,
                                rollout_protocol=ROLLOUT_PROTOCOL,
                                num_samples=100, batch_size=args.tsfm_batch_size,
                                sources={str(p):sha(p) for p in [ROOT/data, ROOT/regress, ROOT/tsfm, split]})
                guard = cache_dir/"cas_cache_identity.json"
                if guard.exists():
                    if read(guard) != identity:
                        raise ValueError(f"Different model/data/protocol in cache: {cache_dir}; use a new cache root")
                elif any(cache_dir.glob("*.parquet")):
                    raise ValueError(f"Unverified preexisting CAS cache: {cache_dir}")
                else:
                    write(guard, identity)
            prediction = run_tsfm_rollout(args=pipeline)
            Xtr, ytr, _, Xva, yva, _, Xte, yte, _, meta = build_features(args=pipeline, pred_table=prediction)
            names = list(meta["feature_names"])
            confs = [i for i,n in enumerate(names) if n.endswith("_conf")]
            points = [i for i,n in enumerate(names) if n.startswith("pred_") and not n.endswith(("_conf", "_uncertainty"))]
            if len(confs) != 1 or len(points) != 1:
                raise ValueError(f"Expected exactly one target point/confidence pair; got {names}")
            if not all(len(y) and len(X)==len(y) and np.isfinite(y).all()
                       for X,y in ((Xtr,ytr),(Xva,yva),(Xte,yte))):
                raise ValueError("Invalid or empty prepared split")
            baseline_check = "prediction files not present; split reused, row parity not verified"
            old = args.split_root/market/f"window_{window}"
            for method in ("tsfm_point_C_fixed_asym", "tsfm_point"):
                reference = old/method/"predictions_test.csv"
                if reference.is_file():
                    previous = pd.read_csv(reference)
                    if (len(previous) != len(yte)
                        or not np.allclose(previous.y_true.to_numpy(), yte, rtol=1e-6, atol=1e-7)
                        or not pd.DatetimeIndex(pd.to_datetime(previous.time, utc=True)).equals(
                            pd.DatetimeIndex(pd.to_datetime(meta["te_time"], utc=True)))):
                        raise ValueError(f"Sample identity differs from original results: {reference}")
                    if upstream == "chronos2" and "tsfm_point" in previous and not np.allclose(previous.tsfm_point, Xte[:,points[0]], rtol=1e-6, atol=1e-7):
                        raise ValueError(f"Chronos predictions differ from original results: {reference}")
                    baseline_check = f"verified test row order, targets and times against {reference}; point parity checked only for original chronos2"
                    break
            point = np.asarray(Xtr[:, points[0]], dtype=float)
            if not np.isfinite(point).all() or not np.isfinite(Xtr[:,confs[0]]).all():
                raise ValueError("Incomplete probability point/confidence on training samples")
            for X in (Xva, Xte):
                if not np.isfinite(X[:, [points[0], confs[0]]]).all():
                    raise ValueError("Incomplete probability features on validation/test; refusing model-dependent sample dropping")
            direction = 1 if np.median(ytr-point) >= 0 else -1
            arrays = folder/"features.npz"
            np.savez(arrays, X_train=Xtr, y_train=ytr, X_valid=Xva, y_valid=yva, X_test=Xte, y_test=yte)
            keys = meta.get("te_keys")
            if keys is not None:
                keys.to_csv(folder/"test_sample_keys.csv", index=False)
            try:
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = "unknown"
            manifest = dict(market=market, window=window, feature_names=names, upstream=upstream,
                            rollout_protocol=ROLLOUT_PROTOCOL if upstream.startswith("cas_") else "timestamp_searchsorted_native",
                            distribution_protocol="cas_common_q10_q90_v1" if upstream.startswith("cas_") else "original_chronos2_native",
                            confidence_index=confs[0], point_index=points[0], direction=direction,
                            counts=dict(train=len(ytr), valid=len(yva), test=len(yte)),
                            train_time=list(meta["tr_time"]), valid_time=list(meta["va_time"]), test_time=list(meta["te_time"]),
                            arrays_sha256=sha(arrays), git_revision=revision, pipeline_args=vars(pipeline),
                            source_sha256={str(p):sha(p) for p in [ROOT/data, ROOT/regress, ROOT/tsfm, split]},
                            chronos_cache_sha256=sha(_get_cache_path(pipeline)),
                            baseline_check=baseline_check,
                            target_scaled=bool(meta.get("target_scaled", False)))
            write(folder/"manifest.json", manifest)
            print(f"PREPARED {market} W{window} counts={manifest['counts']}", flush=True)


def worker(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import pandas as pd
    import torch
    from exp.pipeline.mlp_regressor import MLPRegressor
    folder = args.prepared_root/args.market/f"window_{args.window}"
    manifest = read(folder/"manifest.json")
    arrays = folder/"features.npz"
    if sha(arrays) != manifest["arrays_sha256"]:
        raise ValueError("Prepared feature fingerprint mismatch")
    with np.load(arrays, allow_pickle=False) as data:
        Xtr, ytr, Xva, yva, Xte, yte = [data[k] for k in ["X_train","y_train","X_valid","y_valid","X_test","y_test"]]
    include_c, asymmetric, weighted = METHODS[args.method]
    conf_idx = manifest["confidence_index"]
    weights = training_weights(Xtr[:,conf_idx], args.method, args.weight_lambda, args.seed)
    names = manifest["feature_names"]
    keep = [i for i in range(len(names)) if include_c or i != conf_idx]
    point_test = Xte[:,manifest["point_index"]].copy()
    conf_test = Xte[:,conf_idx].copy()
    Xtr, Xva, Xte = Xtr[:,keep], Xva[:,keep], Xte[:,keep]
    output = args.output_root/args.market/f"window_{args.window}"/args.method/f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = job_config(args, manifest, args.method, args.seed)
    write(output/"run_config.json", config)
    if args.regressor == "mlp":
        model = MLPRegressor(hidden=args.hidden, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                             patience=args.patience, seed=args.seed, device=args.device, threads=args.threads,
                             alpha=args.alpha, direction=manifest["direction"], asymmetric=asymmetric, selection=args.selection)
    else:
        from exp.pipeline.cas_lgbm_regressor import CASLGBMRegressor
        model = CASLGBMRegressor(seed=args.seed, threads=args.threads, alpha=args.alpha,
                                direction=manifest["direction"], asymmetric=asymmetric, selection=args.selection,
                                rounds=args.lgbm_rounds, patience=args.lgbm_patience)
    started = time.monotonic()
    model.fit(Xtr, ytr, Xva, yva, weights)
    model.save(output/("model.pt" if args.regressor == "mlp" else "model.txt"))
    pd.DataFrame(model.history).to_csv(output/"history.csv", index=False)
    metrics = dict(market=args.market, window=args.window, method=args.method, seed=args.seed,
                   upstream=manifest.get("upstream", "chronos2"), regressor=args.regressor,
                   selection=args.selection, best_epoch=model.best_iteration,
                   device=str(model.device) if args.regressor == "mlp" else "cpu",
                   torch_version=torch.__version__, prepared_sha256=manifest["arrays_sha256"])
    for split, X, y in (("valid",Xva,yva),("test",Xte,yte)):
        pred = model.predict(X)
        error = pred-y
        metrics[f"{split}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{split}_rmse"] = float(np.sqrt(np.mean(error**2)))
        frame = pd.DataFrame(dict(time=manifest[f"{split}_time"], y_true=y, prediction=pred))
        if args.regressor == "mlp":
            frame["mlp_prediction"] = pred
        if split == "test":
            frame["tsfm_point"] = point_test; frame["confidence"] = conf_test
        frame.to_csv(output/f"predictions_{split}.csv", index=False)
    metrics["seconds"] = time.monotonic()-started
    write(output/"metrics.json", metrics)  # completion marker written LAST
    print(json.dumps(metrics), flush=True)


def run(args):
    # Independent market/window/seed jobs, one subprocess per explicitly chosen
    # GPU. No DDP: small tabular MLPs benefit more from task parallelism.
    manifests = {}
    for market in args.markets:
        for window in args.windows:
            if not (args.prepared_root/market/f"window_{window}"/"manifest.json").is_file():
                raise FileNotFoundError(f"Prepare first: {market} W{window}")
            manifests[market,window] = read(args.prepared_root/market/f"window_{window}"/"manifest.json")
    jobs = [(m,w,method,s) for m in args.markets for w in args.windows
            for s in args.seeds for method in suite_methods(args.suite)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected = dict(markets=args.markets, windows=args.windows, seeds=args.seeds,
                    methods=suite_methods(args.suite), selection=args.selection)
    expected_path = args.output_root/"expected_jobs.json"
    if expected_path.exists() and read(expected_path) != expected:
        raise ValueError("Output root belongs to a different experiment. Choose a new --output-root.")
    write(expected_path, expected)
    pending = []
    for m,w,method,s in jobs:
        dest = args.output_root/m/f"window_{w}"/method/f"seed_{s}"
        config_path = dest/"run_config.json"
        required = job_config(args, manifests[m,w], method, s)
        if config_path.exists() and read(config_path) != required:
            raise ValueError(f"Different settings in {dest}; choose a new output root")
        if (dest/"metrics.json").exists():
            if not args.resume or not config_path.exists():
                raise FileExistsError(f"Completed result exists: {dest}. Use --resume or a new output root.")
            print(f"SKIP completed {dest}", flush=True)
        else:
            pending.append((m,w,method,s))
    jobs = pending
    logs = args.output_root/"logs"; logs.mkdir(exist_ok=True)
    active, available, failures = [], list(args.gpus), []
    if len(set(available)) != len(available) or ("cpu" in available and len(available)>1):
        raise ValueError("GPU IDs must be unique; cpu must be used alone")
    while jobs or active:
        while jobs and available:
            gpu = available.pop(0)
            m,w,method,s = jobs.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--prepared-root", str(args.prepared_root),
                   "--output-root", str(args.output_root), "--market", m, "--window", str(w), "--method", method,
                   "--seed", str(s), "--device", "cpu" if gpu=="cpu" else "cuda:0",
                   "--alpha",str(args.alpha),"--weight-lambda",str(args.weight_lambda),
                   "--lr",str(args.lr),"--batch-size",str(args.batch_size),"--epochs",str(args.epochs),
                   "--patience",str(args.patience),"--selection",args.selection,"--threads",str(args.threads),
                   "--regressor", args.regressor, "--lgbm-rounds", str(args.lgbm_rounds),
                   "--lgbm-patience", str(args.lgbm_patience),
                   "--hidden", *map(str,args.hidden)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="" if gpu=="cpu" else gpu,
                       OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
            log = open(logs/f"{m}_w{w}_{method}_seed{s}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((proc,gpu,log,(m,w,method,s)))
            print(f"START gpu={gpu} pid={proc.pid} job={(m,w,method,s)}", flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            for proc,_,log,_ in active:
                proc.terminate(); proc.wait(); log.close()
            raise
        for entry in active[:]:
            proc,gpu,log,job = entry
            code = proc.poll()
            if code is not None:
                log.close(); available.append(gpu); active.remove(entry)
                print(f"DONE code={code} job={job}", flush=True)
                if code:
                    failures.append(job)
    summarize(args)
    if failures:
        raise RuntimeError(f"Failed jobs (see logs): {failures}")


def summarize(args):
    import pandas as pd
    rows = [read(p) for p in args.output_root.glob("*/window_*/*/seed_*/metrics.json")]
    if not rows:
        raise ValueError("No completed results")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root/"per_run.csv", index=False)
    expected_path = args.output_root/"expected_jobs.json"
    if expected_path.exists():
        expected = read(expected_path)
        planned = {(m,w,t,s) for m in expected["markets"] for w in expected["windows"]
                   for t in expected["methods"] for s in expected["seeds"]}
        actual = set(zip(frame.market, frame.window, frame.method, frame.seed))
        missing = sorted(planned-actual)
        write(args.output_root/"completion.json", dict(expected=len(planned), completed=len(actual & planned), missing=missing))
        if missing:
            for name in ("per_seed.csv", "summary.csv", "paired_cas_vs_fixed.csv", "paired_by_window.csv"):
                (args.output_root/name).unlink(missing_ok=True)
            print(f"PARTIAL: {len(missing)} jobs missing; per_run.csv written, aggregate tables withheld", flush=True)
            return
    # Average windows WITHIN seed first; std below measures variation over seeds.
    per_seed = frame.groupby(["market","method","selection","seed"], as_index=False).agg(
        mae=("test_mae","mean"), rmse=("test_rmse","mean"), windows=("window","nunique"))
    per_seed.to_csv(args.output_root/"per_seed.csv", index=False)
    report = per_seed.groupby(["market","method","selection"], as_index=False).agg(
        mae_mean=("mae","mean"),mae_std=("mae","std"),rmse_mean=("rmse","mean"),rmse_std=("rmse","std"),seeds=("seed","nunique"))
    report.to_csv(args.output_root/"summary.csv", index=False)
    fixed = per_seed[per_seed.method=="tsfm_point_C_fixed_asym"]
    cas = per_seed[per_seed.method=="tsfm_point_C_confidence_asym"]
    paired = cas.merge(fixed, on=["market","selection","seed"], suffixes=("_cas","_fixed"))
    paired["rmse_change_pct"] = 100*(paired.rmse_cas/paired.rmse_fixed-1)
    paired.to_csv(args.output_root/"paired_cas_vs_fixed.csv", index=False)
    # Keep all windows visible: an average can hide a single-window benefit.
    reference = frame[frame.method == "tsfm_point_C_confidence_asym"]
    pairs = reference.merge(frame, on=["market", "window", "selection", "seed"],
                            suffixes=("_cas", "_control"))
    pairs = pairs[pairs.method_control != "tsfm_point_C_confidence_asym"].copy()
    pairs["rmse_change_pct"] = 100*(pairs.test_rmse_cas/pairs.test_rmse_control-1)
    pairs["mae_change_pct"] = 100*(pairs.test_mae_cas/pairs.test_mae_control-1)
    pairs.to_csv(args.output_root/"paired_by_window.csv", index=False)
    print(report.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare","run","worker","summarize"])
    parser.add_argument("--prepared-root", type=Path, default=Path("results/mlp_transfer_prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("results/mlp_transfer"))
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--upstream", choices=["chronos2", "cas_chronos2", "cas_moirai2", "cas_sundial"], default="chronos2")
    parser.add_argument("--regressor", choices=["mlp", "lightgbm"], default="mlp")
    parser.add_argument("--lgbm-rounds", type=int, default=2000)
    parser.add_argument("--lgbm-patience", type=int, default=100)
    parser.add_argument("--markets", nargs="+", choices=["guangdong","reale_delu","shanxi","reale_fr"], default=["guangdong","reale_delu"])
    parser.add_argument("--windows", nargs="+", type=int, choices=[1,2,3], default=[1,2,3])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    parser.add_argument("--suite", choices=["core","original","mechanism"], default="original")
    parser.add_argument("--resume", action="store_true", help="skip only completed jobs with matching settings")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tsfm-batch-size", type=int, default=16)
    parser.add_argument("--hidden", nargs="+", type=int, default=[128,64])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--weight-lambda", type=float, default=1)
    parser.add_argument("--selection", choices=["legacy","rmse"], default="legacy")
    parser.add_argument("--market")
    parser.add_argument("--window", type=int)
    parser.add_argument("--method", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = {"chronos2": os.environ.get("CHRONOS2_MODEL_PATH", "models/chronos-2"),
                           "cas_chronos2": os.environ.get("CHRONOS2_MODEL_PATH", "models/chronos-2"),
                           "cas_moirai2": "Salesforce/moirai-2.0-R-small",
                           "cas_sundial": "thuml/sundial-base-128m"}[args.upstream]
    import math
    if not math.isfinite(args.weight_lambda) or args.weight_lambda < 0:
        parser.error("--weight-lambda must be finite and nonnegative")
    if args.command == "prepare" and (args.split_root is None or args.cache_root is None):
        parser.error("prepare requires --split-root and --cache-root from the original experiment")
    if args.lgbm_rounds < 1 or args.lgbm_patience < 1:
        parser.error("LightGBM rounds and patience must be positive")
    if args.command == "worker" and any(getattr(args,k) is None for k in ("market","window","method")):
        parser.error("worker requires --market --window --method")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
