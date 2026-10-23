#!/bin/bash
# Run the original PBO on the hold-out problems (envs generated from pboplus.benchmarks).
# Hyper-parameters: those of the original rosenbrock_5d.json (not tuned per problem).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N_AVG=${N_AVG:-5}
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=$ROOT/baselines/pbo_original
for env in "$@"; do
  d=$ROOT/bench/original/$env; rm -rf $d; mkdir -p $d; cd $d
  cat > $env.py <<PY
import sys, numpy as np
sys.path.insert(0, "$ROOT")
from pboplus.registry import get
class $env:
    def __init__(self, path):
        self.p = get("$env")
        self.name, self.act_size = "$env", self.p.dim
        self.obs_size = self.act_size
        self.obs = np.zeros(self.obs_size)
    def observe(self):
        return self.obs
    def step(self, actions, ep):
        return -self.p(np.asarray(actions)), list(self.p.convert(np.asarray(actions)))
    def close(self):
        pass
PY
  python3 - <<PY
import json, sys
sys.path.insert(0, "$ROOT")
from pboplus.registry import get
pr = get("$env")
p = json.load(open("$ROOT/baselines/pbo_original/envs/rosenbrock_5d.json"))
p.update(env_name="$env", n_avg=$N_AVG, n_gen=pr.n_gen, n_ind=pr.n_ind, n_cpu=pr.n_ind, avg_type="linear")
json.dump(p, open("$env.json", "w"), indent=1)
PY
  python3 -c "import sys;sys.argv=['pbo','$env.json'];from pbo.src.run import run;run()" > log.txt 2>&1
  echo "done $env"
done
