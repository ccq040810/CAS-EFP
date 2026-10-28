"""Train only 18 loss-only CAS jobs; verify and reuse 36 existing no-c controls."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mlp_transfer import job_config, read, write
from scripts.run_cas_all import prepared_manifest, run_lock

METHOD = "tsfm_point_confidence_asym_noC"
CONTROLS = ("tsfm_point", "tsfm_point_fixed_asym_noC")


def checked_result(folder, required, market, window, method, seed):
    if read(folder/"run_config.json") != required:
        raise ValueError(f"Training configuration differs: {folder}")
    metrics = read(folder/"metrics.json")
    expected = dict(market=market, window=window, method=method, seed=seed,
                    upstream="cas_chronos2", regressor="lightgbm", selection="rmse",
                    prepared_sha256=required["prepared_sha256"])
    if any(metrics.get(k) != v for k,v in expected.items()):
        raise ValueError(f"Metric identity differs: {folder}")
    import math
    if not all(math.isfinite(metrics[k]) for k in ("test_mae", "test_rmse", "valid_mae", "valid_rmse")):
        raise ValueError(f"Nonfinite metrics: {folder}")
    if any(n.endswith("_conf") for n in required["feature_names"]):
        raise ValueError("Confidence must not enter the regressor")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=ROOT/"results/cas_transfer_prepared_v2/cas_chronos2")
    parser.add_argument("--control-root", type=Path, default=ROOT/"results/cas_all_20261028_v2/cas_chronos2/lightgbm")
    parser.add_argument("--output-root", type=Path, default=ROOT/"results/cas_loss_only_20261028")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    for key in ("prepared_root", "control_root", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_root == args.control_root or args.control_root in args.output_root.parents:
        parser.error("Use a separate output root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(regressor="lightgbm", hidden=[128,64], lr=.001,
        batch_size=8192, epochs=300, patience=30, selection="rmse", alpha=2.,
        weight_lambda=1., lgbm_rounds=2000, lgbm_patience=100)
    with run_lock(args.output_root):
        jobs, rows = [], []
        for market in ("guangdong", "reale_delu"):
            for window in (1,2,3):
                manifest = prepared_manifest(args.prepared_root/market/f"window_{window}", "cas_chronos2")
                for seed in (42,43,44):
                    for method in CONTROLS:
                        folder = args.control_root/market/f"window_{window}"/method/f"seed_{seed}"
                        required = job_config(settings, manifest, method, seed)
                        rows.append(checked_result(folder, required, market, window, method, seed))
                    jobs.append((market, window, seed, job_config(settings, manifest, METHOD, seed)))
        config = dict(prepared_root=str(args.prepared_root), control_root=str(args.control_root),
                      methods=[*CONTROLS,METHOD], selection="rmse", alpha=2., weight_lambda=1.,
                      markets=["guangdong","reale_delu"], windows=[1,2,3], seeds=[42,43,44])
        path = args.output_root/"experiment.json"
        if path.exists() and read(path) != config:
            raise ValueError("Different experiment in output root")
        write(path, config)
        print("READY: 6 verified v2 feature sets; 36 matching no-c controls; 18 new CPU jobs", flush=True)
        if args.check_only:
            return
        for name in ("FINAL_RESULTS.csv", "PAIRED_BY_WINDOW.csv", "RESULTS.md"):
            (args.output_root/name).unlink(missing_ok=True)
        write(args.output_root/"completion.json", dict(expected_new=18, completed_new=0, reused_controls=36, complete=False))
        logs = args.output_root/"logs"
        logs.mkdir(exist_ok=True)
        def train(job):
            market, window, seed, required = job
            folder = args.output_root/market/f"window_{window}"/METHOD/f"seed_{seed}"
            if (folder/"run_config.json").exists() and read(folder/"run_config.json") != required:
                raise ValueError(f"Different settings: {folder}")
            if (folder/"metrics.json").exists():
                return checked_result(folder, required, market, window, METHOD, seed)
            cmd = [sys.executable, str(ROOT/"scripts/run_mlp_transfer.py"), "worker",
                "--prepared-root", str(args.prepared_root), "--output-root", str(args.output_root),
                "--regressor", "lightgbm", "--market", market, "--window", str(window),
                "--method", METHOD, "--seed", str(seed), "--selection", "rmse",
                "--device", "cpu", "--threads", str(args.threads)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            print(f"TRAIN {market} W{window} seed{seed}", flush=True)
            with (logs/f"{market}_w{window}_s{seed}.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            return checked_result(folder, required, market, window, METHOD, seed)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows.extend(pool.map(train, jobs))
        import pandas as pd
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_root/"ALL_RUNS.csv", index=False)
        per_seed = frame.groupby(["market","method","seed"], as_index=False).agg(mae=("test_mae","mean"), rmse=("test_rmse","mean"))
        per_seed.to_csv(args.output_root/"PER_SEED.csv", index=False)
        report = per_seed.groupby(["market","method"], as_index=False).agg(mae_mean=("mae","mean"), mae_std=("mae","std"), rmse_mean=("rmse","mean"), rmse_std=("rmse","std"))
        report.to_csv(args.output_root/"FINAL_RESULTS.csv", index=False)
        paired = frame[frame.method==METHOD].merge(frame[frame.method.isin(CONTROLS)], on=["market","window","seed"], suffixes=("_cas","_control"))
        paired["rmse_change_pct"] = 100*(paired.test_rmse_cas/paired.test_rmse_control-1)
        paired["mae_change_pct"] = 100*(paired.test_mae_cas/paired.test_mae_control-1)
        paired.to_csv(args.output_root/"PAIRED_BY_WINDOW.csv", index=False)
        text = "# CAS仅损失加权：Chronos-2 + LightGBM\n\n18次新增训练，36次核验后复用对照；三个方案均不输入c。保持v2历史处理和普通验证RMSE早停，仅改变训练损失。\n\n各随机种子先平均三个窗口，再计算种子均值与样本标准差。MAE/RMSE原单位：广东元/MWh、德卢欧元/MWh；除1000转换为对应币种/kWh。百分比为CAS/对照−1，负值表示改善。\n\n"
        text += "|市场|方案|MAE均值|MAE标准差|RMSE均值|RMSE标准差|\n|---|---|---:|---:|---:|---:|\n"
        for row in report.itertuples():
            text += f"|{row.market}|{row.method}|{row.mae_mean:.6f}|{row.mae_std:.6f}|{row.rmse_mean:.6f}|{row.rmse_std:.6f}|\n"
        (args.output_root/"RESULTS.md").write_text(text, encoding="utf-8")
        write(args.output_root/"completion.json", dict(expected_new=18, completed_new=18, reused_controls=36, complete=True))
        print(report.to_string(index=False), flush=True)
        print(f"COMPLETE: {args.output_root/'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
