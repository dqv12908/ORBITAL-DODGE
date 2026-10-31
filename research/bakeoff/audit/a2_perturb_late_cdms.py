"""Audit 2: perturb every CDM with ttca < 2 d (all numeric cols incl. risk/miss, but keep ttca/event/mission so eligibility is unchanged)
and rebuild features/sequences with the harness recipe. Any change => late-CDM leakage. Also explains the 5826 d1/dall quirk."""
import sys, json
sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd
import harness as H
BK = H.HERE
tr = H.clean(H.load_raw("train"))
late = tr.time_to_tca < 2
num = [c for c in tr.columns if c not in ("event_id", "mission_id", "time_to_tca") and tr[c].dtype.kind in "if"]
rng = np.random.default_rng(1)
tp = tr.copy(); tp[num] = tp[num].astype(float)
tp.loc[late, num] = rng.normal(size=(late.sum(), len(num))) * 1e3
inp, tgt = H.eligible_frame(tp); eids = tgt.index.values
numcols = [c for c in tr.columns if c not in ("event_id", "mission_id") and tr[c].dtype.kind in "if"]
out = {"n_late_rows_perturbed": int(late.sum())}
for name, (cl, tc) in {"A13": (H.ALLOWED[1:], H.TREND_COLS_A13), "ALL": (numcols, H.TREND_COLS_ALL)}.items():
    F = H.event_features(inp, cl, tc).loc[eids]; F.index = [f"train:{e}" for e in eids]; F.columns = [str(c) for c in F.columns]
    G = pd.read_parquet(f"{BK}/data/features_{name}.parquet")
    same = np.allclose(F[G.columns].values.astype(float), G.values, equal_nan=True, rtol=0, atol=0)
    out[f"{name}_identical_after_perturbing_late_cdms"] = bool(same)
X, M = H.build_sequences(inp, eids)
S = H.load_sequences()
out["sequences_identical"] = bool(np.array_equal(np.nan_to_num(X, nan=-9e9), np.nan_to_num(S["X"], nan=-9e9)) and np.array_equal(M, S["mask"]))
# 5826 quirk
d = H.clean(H.load_raw("train")); d = d[(d.event_id == 5826) & (d.time_to_tca >= 2)].sort_values("time_to_tca", ascending=False)
out["ev5826_c_sigma_r_seq"] = d.c_sigma_r.tolist(); out["ev5826_ttca"] = d.time_to_tca.round(3).tolist()
print(json.dumps(out, indent=1)); json.dump(out, open(f"{BK}/audit/a2_out.json", "w"), indent=1)
