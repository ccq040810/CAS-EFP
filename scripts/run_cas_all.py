"""One command: check real environments, prepare 18 windows, train 756 jobs, report.

Uses existing efp for Chronos and downstream training, isolated environments for
Moirai/Sundial. No dependency installation or model download is performed here.
Rerunning the same command resumes only matching, verified completed artifacts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_mlp_transfer import MECHANISM as METHODS, ROLLOUT_PROTOCOL, job_config, read, sha, summarize, write
from scripts.run_cas_transfer_matrix import verify_alignment


@contextmanager
def run_lock(folder):
    """OS-held lock releases on process exit, including a crash; no stale PID lock."""
    import fcntl
    path = Path(folder)/"launcher.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another launcher is already running in this output root")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def expected_jobs(sources, markets, windows, seeds):
    return [(source,backend,market,window,method,seed)
            for source in sources for backend in ("mlp","lightgbm")
            for market in markets for window in windows for seed in seeds for method in METHODS]


def prepared_manifest(folder, source):
    manifest = read(Path(folder)/"manifest.json")
    if manifest.get("upstream") != source or manifest.get("distribution_protocol") != "cas_common_q10_q90_v1":
        raise ValueError(f"Wrong probability source/protocol in {folder}")
    if manifest.get("rollout_protocol") != ROLLOUT_PROTOCOL:
        raise ValueError(f"Pre-fix history/cache protocol in {folder}; prepare into a new v2 root")
    if sha(Path(folder)/"features.npz") != manifest["arrays_sha256"]:
        raise ValueError(f"Feature fingerprint mismatch in {folder}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root",type=Path,default=ROOT.parent/"CAS-EFP")
    parser.add_argument("--bundle-root",type=Path,default=ROOT.parent/"cas-offline-bundle-20261027")
    parser.add_argument("--env-root",type=Path,default=ROOT.parent/"cas-envs")
    parser.add_argument("--prepared-root",type=Path,default=ROOT/"results/cas_transfer_prepared_v2")
    parser.add_argument("--cache-root",type=Path,default=ROOT/"results/cas_transfer_cache_v2")
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--selection",choices=["rmse","legacy"],default="rmse",
                        help="legacy restores the original asymmetric validation early stopping; not the old history bug")
    parser.add_argument("--gpus",nargs="+",default=list(map(str,range(8))))
    parser.add_argument("--cpu-workers",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--phase",choices=["all","check","prepare","train","summarize"],default="all")
    args = parser.parse_args()
    if args.output_root is None:
        suffix = "_legacy_selection" if args.selection == "legacy" else ""
        args.output_root = ROOT/f"results/cas_all_20261028_v2{suffix}"
    if len(set(args.gpus)) != len(args.gpus) or args.cpu_workers < 1 or args.threads < 1:
        parser.error("Unique GPU IDs and positive CPU settings are required")
    for key in ("old_root","bundle_root","env_root","prepared_root","cache_root","output_root"):
        setattr(args,key,getattr(args,key).resolve())
    args.output_root.mkdir(parents=True,exist_ok=True)
    sources = {
        "cas_chronos2": (sys.executable, args.old_root/"models/chronos-2"),
        "cas_moirai2": (str(args.env_root/"moirai-py312/bin/python"),args.bundle_root/"models/moirai-2.0-R-small"),
        "cas_sundial": (str(args.env_root/"sundial-py312/bin/python"),args.bundle_root/"models/sundial-base-128m"),
    }
    markets, windows, seeds = ["guangdong","reale_delu"], [1,2,3], [42,43,44]
    config = dict(sources={k:list(map(str,v)) for k,v in sources.items()},markets=markets,windows=windows,
                  seeds=seeds,methods=list(METHODS),selection=args.selection,alpha=2.,weight_lambda=1.,
                  hidden=[128,64],lr=.001,batch_size=8192,epochs=300,patience=30,
                  lgbm_rounds=2000,lgbm_patience=100,threads=args.threads,
                  prepared_root=str(args.prepared_root),cache_root=str(args.cache_root),
                  old_root=str(args.old_root),rollout_protocol=ROLLOUT_PROTOCOL)
    logs = args.output_root/"logs"; logs.mkdir(exist_ok=True)
    cards = queue.Queue()
    for card in args.gpus:
        cards.put(card)
    active, mutex = set(), threading.RLock()
    stopping = threading.Event()

    def status(*values):
        with mutex:
            print(" ".join(map(str,values)),flush=True)

    def interrupt(signum, frame):
        stopping.set()
        with mutex:
            for process in active:
                process.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)

    def execute(cmd, logfile, gpu=None, offline=True):
        if stopping.is_set():
            raise RuntimeError("Launcher interrupted; no new subprocess started")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
                   CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1")
        if offline:
            env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
        with Path(logfile).open("w") as log:
            process = subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            with mutex:
                active.add(process)
            try:
                code = process.wait()
                if code:
                    raise RuntimeError(f"Exit {code}; see {logfile}")
            finally:
                with mutex:
                    active.discard(process)

    def pool_jobs(jobs, fn, workers):
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn,j):j for j in jobs}
            try:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(dict(job=list(futures[future]),error=str(exc)))
                        status("FAILED",futures[future],str(exc))
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                with mutex:
                    for process in active:
                        process.terminate()
                raise
        return failures

    with run_lock(args.output_root):
        config_path = args.output_root/"experiment.json"
        if config_path.exists() and read(config_path) != config:
            raise ValueError("Different experiment settings; choose a new --output-root")
        write(config_path,config)
        if args.phase in ("all","check","prepare","train"):
            cmd = [sys.executable,ROOT/"scripts/check_cas_environment.py","--old-root",args.old_root,
                   "--output-root",args.output_root/"environment","--expected-gpus",str(len(args.gpus))]
            for source,(python,model) in sources.items():
                family = source[4:]
                cmd += [f"--{family}-python",python,f"--{family}-path",model]
            status("CHECK all real model environments")
            execute(cmd,logs/"environment.log",gpu=",".join(args.gpus))
            if args.phase == "check":
                return

        if args.phase in ("all","prepare"):
            def prepare(job):
                source,market,window = job
                folder = args.prepared_root/source/market/f"window_{window}"
                if (folder/"manifest.json").is_file():
                    meta = prepared_manifest(folder,source)
                    for path,digest in meta["source_sha256"].items():
                        if sha(path) != digest:
                            raise ValueError(f"Prepared input changed: {path}")
                    status("SKIP prepared",source,market,window)
                    return
                # Never delete an interrupted user's folder. Prepare in a new staging tree.
                staging = args.prepared_root/".staging"/uuid.uuid4().hex
                gpu = cards.get()
                try:
                    python,model = sources[source]
                    cmd = [python,ROOT/"scripts/run_mlp_transfer.py","prepare","--upstream",source,
                           "--model-path",model,"--split-root",args.old_root/"results/rolling_adaptive_v3",
                           "--cache-root",args.cache_root,"--prepared-root",staging,
                           "--markets",market,"--windows",str(window),"--device","cuda:0"]
                    status("PREPARE",gpu,source,market,window)
                    execute(cmd,logs/f"prepare_{source}_{market}_w{window}.log",gpu)
                    ready = staging/market/f"window_{window}"
                    prepared_manifest(ready,source)
                    folder.parent.mkdir(parents=True,exist_ok=True)
                    if folder.exists():
                        folder.rename(folder.with_name(folder.name+"_interrupted_"+uuid.uuid4().hex[:8]))
                    ready.rename(folder)
                finally:
                    cards.put(gpu)
            failures = pool_jobs([(s,m,w) for s in sources for m in markets for w in windows],prepare,len(args.gpus))
            if failures:
                write(args.output_root/"preparation_failures.json",failures)
                raise RuntimeError("Some preparations failed; rerun after fixing their logs")

        verify_alignment(dict(upstreams=sources,markets=markets,windows=windows,prepared_root=str(args.prepared_root)))
        manifests = {(s,m,w):prepared_manifest(args.prepared_root/s/m/f"window_{w}",s)
                     for s in sources for m in markets for w in windows}
        jobs = expected_jobs(sources,markets,windows,seeds)
        for source in sources:
            for backend in ("mlp","lightgbm"):
                out = args.output_root/source/backend; out.mkdir(parents=True,exist_ok=True)
                write(out/"expected_jobs.json",dict(markets=markets,windows=windows,seeds=seeds,
                                                  methods=list(METHODS),selection=args.selection))
        if args.phase in ("all","train"):
            defaults = dict(hidden=[128,64],lr=.001,batch_size=8192,epochs=300,patience=30,selection=args.selection,
                            alpha=2.,weight_lambda=1.,lgbm_rounds=2000,lgbm_patience=100)
            def train(job):
                source,backend,market,window,method,seed = job
                dest = args.output_root/source/backend/market/f"window_{window}"/method/f"seed_{seed}"
                settings = job_config(SimpleNamespace(**defaults,regressor=backend),manifests[source,market,window],method,seed)
                if (dest/"run_config.json").exists() and read(dest/"run_config.json") != settings:
                    raise ValueError(f"Different training settings: {dest}")
                if (dest/"metrics.json").exists():
                    if not (dest/"run_config.json").is_file():
                        raise ValueError(f"Unverified completion marker: {dest}")
                    status("SKIP trained",job)
                    return
                gpu = cards.get() if backend == "mlp" else "cpu"
                try:
                    cmd = [sys.executable,ROOT/"scripts/run_mlp_transfer.py","worker",
                           "--prepared-root",args.prepared_root/source,"--output-root",args.output_root/source/backend,
                           "--regressor",backend,"--market",market,"--window",str(window),"--method",method,
                           "--seed",str(seed),"--selection",args.selection,"--threads",str(args.threads),
                           "--device","cuda:0" if backend == "mlp" else "cpu"]
                    status("TRAIN",gpu,job)
                    execute(cmd,logs/f"{source}_{backend}_{market}_w{window}_{method}_s{seed}.log",gpu)
                finally:
                    if backend == "mlp":
                        cards.put(gpu)
            with ThreadPoolExecutor(max_workers=2) as coordinator:
                futures = [coordinator.submit(pool_jobs,[j for j in jobs if j[1]==backend],train,workers)
                           for backend,workers in (("mlp",len(args.gpus)),("lightgbm",args.cpu_workers))]
                failures = [error for future in futures for error in future.result()]
            write(args.output_root/"training_failures.json",failures)

        import pandas as pd
        reports, comparisons, missing = [], [], []
        for source in sources:
            for backend in ("mlp","lightgbm"):
                out = args.output_root/source/backend
                if not list(out.glob("*/window_*/*/seed_*/metrics.json")):
                    missing.append([source,backend,"no completed jobs"])
                    continue
                summarize(SimpleNamespace(output_root=out))
                complete = read(out/"completion.json")
                missing.extend([source,backend,*row] for row in complete["missing"])
                if not complete["missing"]:
                    reports.append(pd.read_csv(out/"summary.csv").assign(upstream=source,regressor=backend))
                    comparisons.append(pd.read_csv(out/"paired_by_window.csv").assign(upstream=source,regressor=backend))
        completed = sum((args.output_root/s/b/m/f"window_{w}"/t/f"seed_{seed}/metrics.json").is_file()
                        for s,b,m,w,t,seed in jobs)
        write(args.output_root/"completion.json",dict(expected=len(jobs),completed=completed,missing=missing))
        if missing:
            for name in ("FINAL_RESULTS.csv","FINAL_PAIRED_BY_WINDOW.csv","RESULTS.md"):
                (args.output_root/name).unlink(missing_ok=True)
            raise RuntimeError(f"Incomplete {completed}/{len(jobs)}; see completion.json and logs")
        table = pd.concat(reports,ignore_index=True)
        table.to_csv(args.output_root/"FINAL_RESULTS.csv",index=False)
        pd.concat(comparisons,ignore_index=True).to_csv(args.output_root/"FINAL_PAIRED_BY_WINDOW.csv",index=False)
        lines = ["# CAS model transfer results","",f"Completed {completed}/{len(jobs)} planned training jobs.",
                 "","Units follow original input data; no MWh-to-kWh conversion is applied.",
                 "Seed standard deviations are not confidence intervals.","",
                 "| Upstream | Regressor | Market | Method | MAE mean | RMSE mean | RMSE seed SD |",
                 "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for row in table.itertuples():
            lines.append(f"| {row.upstream} | {row.regressor} | {row.market} | {row.method} | {row.mae_mean:.8g} | {row.rmse_mean:.8g} | {row.rmse_std:.5g} |")
        (args.output_root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        print("ALL COMPLETE",completed,"jobs:",args.output_root/"FINAL_RESULTS.csv",flush=True)


if __name__ == "__main__":
    main()
