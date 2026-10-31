"""Audit 5: per-fold AUC under label shuffle (removes between-fold offset that inflates/deflates pooled AUC)."""
import sys, json
sys.path.insert(0, "/root/orbital-dodge/research/bakeoff"); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff/audit")
import numpy as np
from sklearn.metrics import roc_auc_score
import importlib.util
src = open("/root/orbital-dodge/research/bakeoff/audit/a3_lgbm_repro_shuffle.py").read().split("for split in [\"mission\", \"event\"]:\n    t = time.time()")[0]
exec(src)
fe = H.fold_ids("event"); fm = H.fold_ids("mission"); res = {}
for split, f in [("event", fe), ("mission", fm)]:
    per = []
    for s in range(3):
        P = run(split, np.random.default_rng(100 + s).permutation(y), seed=s)
        per.append([round(float(roc_auc_score(yc[f == k], P[f == k, 2])), 3) for k in range(5)])
    res[split] = per; res[split + "_mean_within_fold"] = float(np.mean(per))
print(json.dumps(res)); json.dump(res, open("/root/orbital-dodge/research/bakeoff/audit/a5_out.json", "w"))
