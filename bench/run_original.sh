#!/bin/bash
# Run the original PBO (TF) on each benchmark with N_AVG repetitions.
# Usage: bench/run_original.sh env1 env2 ...
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N_AVG=${N_AVG:-10}
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$ROOT/baselines/pbo_original
for env in "$@"; do
  d=$ROOT/bench/original/$env; rm -rf $d; mkdir -p $d; cd $d
  cp $ROOT/baselines/pbo_original/envs/*.py .
  python3 - <<PY
import json
p=json.load(open("$ROOT/baselines/pbo_original/envs/$env.json"))
p["n_avg"]=$N_AVG; p.setdefault("type","continuous")
json.dump(p,open("$env.json","w"),indent=1)
PY
  python3 -c "import sys;sys.argv=['pbo','$env.json'];from pbo.src.run import run;run()" > log.txt 2>&1
  echo "done $env"
done
