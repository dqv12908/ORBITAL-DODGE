"""Audit 4: runner-up tabpfn: (i) reproduce mission fold 3 (smallest crit count, fast) vs stored OOF, (ii) shuffle-label fit on that fold."""
import sys, json, time, os
sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd, torch
import harness as H
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import roc_auc_score
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; yc = (y == 2).astype(int); mis = ev.mission_id.values
X = H.load_features("A13").loc[keys].values.astype(np.float32); X[~np.isfinite(X)] = np.nan
F = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out = {"fold": F}
for k, trn, tst in H.folds("mission"):
    if k != F: continue
    assert len(np.intersect1d(mis[trn], mis[tst])) == 0
    O = pd.read_parquet(f"{H.OOF}/tabpfn_mission.parquet").set_index("event_key").loc[keys[tst]][H.PCOLS].values
    for tag, ylab in [("true", y), ("shuffled", np.random.default_rng(5).permutation(y))]:
        t = time.time()
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda:0", n_estimators=8, random_state=0)
        m.fit(X[trn], ylab[trn]); P = m.predict_proba(X[tst]); P = P / P.sum(1, keepdims=True)
        out[f"{tag}_auc"] = float(roc_auc_score(yc[tst], P[:, 2])); out[f"{tag}_secs"] = round(time.time() - t, 1)
        if tag == "true": out["repro_maxabsdiff_vs_oof"] = float(np.abs(P - O).max()); out["stored_fold_auc"] = float(roc_auc_score(yc[tst], O[:, 2]))
        print(json.dumps(out), flush=True)
json.dump(out, open(f"/root/orbital-dodge/research/bakeoff/audit/a4_out_fold{F}.json", "w"), indent=1)
