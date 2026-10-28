#!/usr/bin/env bash
# Detach one resumable launcher; all inputs/environments are checked before jobs.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${CAS_BASE_PYTHON:-/share/bupt/czl/miniconda3/envs/efp/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Set CAS_BASE_PYTHON to the original efp Python executable."
  exit 1
fi
mkdir -p results
nohup "$python_bin" -u scripts/run_cas_all.py "$@" \
  >> results/cas_all_launcher.log 2>&1 < /dev/null &
launcher_pid=$!
echo "Launcher PID: $launcher_pid"
echo "Progress: tail -f $repo_root/results/cas_all_launcher.log"
echo "Final: $repo_root/results/cas_all_20261028_v2/RESULTS.md"
echo "Completion: $repo_root/results/cas_all_20261028_v2/completion.json"
