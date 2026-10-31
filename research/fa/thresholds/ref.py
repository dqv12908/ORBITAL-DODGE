# Reproduce current served model (lgbm ALL, organizer pop) with inner OOF, in standard candidate format.
import os, sys, numpy as np, pandas as pd
ORGD = "/root/orbital-dodge/research/organizer"
sys.path.insert(0, ORGD)
import organizer as O
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
OUTD = "/root/orbital-dodge/research/fa/thresholds"
ev, FE = O.load(); X = FE["ALL"]; y = ev.y_tier.values; n = len(ev)
F = pd.read_parquet(os.path.join(O.DATA, "features_ALL.parquet")).loc[ev.event_id]
cot = F["last_c_object_type"].fillna(-1).astype(int).values
outer = []
for s in range(5):
    trn, tst = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(np.arange(n), groups=ev.event_id.values))
    outer.append((f"8020_s{s}", 0, "event80_20", s, np.sort(trn), np.sort(tst)))
for k, (trn, tst) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)):
    outer.append(("mission", k, "mission", k, trn, tst))
jobs = []
for name, fold, sp, s, trn, tst in outer:
    jobs.append((name, fold, "outer_test", trn, tst))
    for ia, ib in O.inner_folds(sp, trn, ev, s):
        jobs.append((name, fold, "inner_oof", trn[ia], trn[ib]))
O.log("fits", len(jobs))
Ps = Parallel(n_jobs=16)(delayed(O.fit_pred)(O.BEST["ALL"], X[a], y[a], X[b]) for _, _, _, a, b in jobs)
rows = []
for (name, fold, role, a, b), P in zip(jobs, Ps):
    rows.append(pd.DataFrame({"event_key": ev.event_id.values[b], "split": name, "fold": fold, "role": role,
        "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2], "y_tier": y[b],
        "latest_tier": ev.latest_tier.values[b], "c_object_type": cot[b],
        "final_sub": ev.subpop.values[b]}))
df = pd.concat(rows, ignore_index=True)
df.to_parquet(os.path.join(OUTD, "current_lgbm_all.parquet"))
O.log("done", df.shape, df.groupby(["split", "role"]).size().to_dict())
