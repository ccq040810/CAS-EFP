"""Prepare each probability source once; fit both regression families on it."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(config, stage, resume=False):
    runner = str(ROOT/"scripts/run_mlp_transfer.py")
    for upstream, model_path in config["upstreams"].items():
        if upstream not in ("cas_chronos2", "cas_moirai2", "cas_sundial"):
            raise ValueError(f"Common-grid probability adapter required: {upstream}")
        prepared = str(Path(config["prepared_root"])/upstream)
        shared = ["--prepared-root", prepared, "--markets", *config["markets"],
                  "--windows", *map(str, config["windows"])]
        if stage == "prepare":
            yield [sys.executable, runner, "prepare", *shared, "--upstream", upstream,
                   "--model-path", model_path, "--split-root", config["split_root"],
                   "--cache-root", config["cache_root"], "--device", config.get("prepare_device", "cuda:0")]
        else:
            for regressor in config["regressors"]:
                output = str(Path(config["output_root"])/upstream/regressor)
                cmd = [sys.executable, runner, stage, *shared, "--output-root", output,
                       "--regressor", regressor, "--selection", config.get("selection", "rmse"),
                       "--suite", config.get("suite", "original"),
                       "--seeds", *map(str, config["seeds"]),
                       "--gpus", *(config.get("gpus", ["0"]) if regressor == "mlp" else ["cpu"])]
                if resume and stage == "run":
                    cmd.append("--resume")
                yield cmd


def verify_alignment(config):
    """Require equal targets, row order, times and non-TSFM features across sources."""
    import numpy as np
    import pandas as pd
    reference = next(iter(config["upstreams"]))
    for market in config["markets"]:
        for window in config["windows"]:
            previous = None
            for source in config["upstreams"]:
                root = Path(config["prepared_root"])/source/market/f"window_{window}"
                meta = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
                keep = [i for i in range(len(meta["feature_names"]))
                        if i not in (meta["point_index"], meta["confidence_index"])]
                with np.load(root/"features.npz", allow_pickle=False) as data:
                    current = {k: data[k].copy() for k in ("y_train", "y_valid", "y_test")}
                    current.update({k: data[k][:, keep].copy() for k in ("X_train", "X_valid", "X_test")})
                current["valid_time"] = np.asarray(meta["valid_time"])
                current["test_time"] = np.asarray(meta["test_time"])
                keyfile = root/"test_sample_keys.csv"
                if not keyfile.is_file():
                    raise ValueError(f"Missing exact forecast sample keys: {keyfile}")
                current["test_keys"] = pd.read_csv(keyfile).astype(str).to_numpy()
                current["feature_names"] = np.asarray([meta["feature_names"][i] for i in keep])
                if meta["distribution_protocol"] != "cas_common_q10_q90_v1":
                    raise ValueError("Different quantile protocol in matrix")
                if previous is not None:
                    for k in current:
                        np.testing.assert_array_equal(previous[k], current[k],
                                                      err_msg=f"{source} vs {reference}: {market} W{window} {k}")
                else:
                    previous = current
    print("Verified identical labels, sample keys, times and non-TSFM features across upstreams", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "summarize", "verify"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "verify":
        verify_alignment(config)
        return
    if args.stage == "run" and not args.dry_run:
        verify_alignment(config)
    for cmd in commands(config, args.stage, args.resume):
        print(json.dumps(cmd, ensure_ascii=False), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    if args.stage == "prepare" and not args.dry_run:
        verify_alignment(config)


if __name__ == "__main__":
    main()
