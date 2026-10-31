"""Train MDN transition models per fold and roll out held-out events.
usage: run_futures.py KIND [n_paths]   KIND in mission|event|lomo|full"""
import sys, os, time, json, joblib, numpy as np, pandas as pd, torch
from bm import *
from mdn import Futures

kind = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
GRID = np.round(np.arange(7.0, -0.001, -0.25), 3)
logf = open(os.path.join(OUT, f"futures_{kind}.log"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

d = load_rows(); ev = events_table(d)
dtr = d[d.split == "train"]
evtr = ev[ev.split == "train"].reset_index(drop=True)
el = evtr[evtr.eligible].reset_index(drop=True)
last_flag = dtr.groupby("event_key").cumcount(ascending=False) == 0
final_ttca = dtr.event_key.map(evtr.set_index("event_key").final_ttca)


def train_on(keys_train, seed=0):
    rows = dtr[dtr.event_key.isin(keys_train)]
    a, Y, F = pairs(rows)
    tr = rows[(rows.time_to_tca < 1.0) & (final_ttca.loc[rows.index] < 1.0)]
    T = last_flag.loc[tr.index].values.astype(np.float32)
    log(f"  pairs {len(a)} term rows {len(tr)} (last frac {T.mean():.3f})")
    fut = Futures().fit(a[XCOLS].values.astype(np.float64), Y, F, tr[XCOLS].values.astype(np.float64), T, seed=seed, log=log)
    return fut


def roll(fut, ev_sub, n_paths, keep=0, seed=0):
    X0 = d.loc[ev_sub.start_idx.values, XCOLS].values.astype(np.float64)
    chunk = max(1, 400000 // n_paths)
    res = []
    for i in range(0, len(X0), chunk):
        res.append(fut.rollout(X0[i:i + chunk], n_paths, GRID, seed=seed + i, keep_paths=keep))
    out = {k: np.concatenate([r[k] for r in res]) for k in ["p_tiers", "q_lm", "q_rk", "n_steps", "final_ttca_med"]}
    if keep:
        out["paths"] = {k: np.concatenate([r["paths"][k] for r in res]) for k in ["T", "L", "R"]}
    out["event_key"] = ev_sub.event_key.values
    return out


t0 = time.time()
if kind == "full":
    fut = train_on(evtr.event_key.values)
    torch.save(fut.state_dict(), os.path.join(OUT, "mdn_full.pt"))
    evte = ev[ev.split == "test"].reset_index(drop=True)
    r = roll(fut, evte, NP); r["val_nll"] = fut.val_nll
    joblib.dump(r, os.path.join(OUT, "futures_full_test.joblib"))
    log("test rollout done", len(evte), f"{time.time()-t0:.0f}s")
    sys.exit()

fl = folds(el, kind)
allres = []
for fi, (tri, tei) in enumerate(fl):
    held = el.iloc[tei]
    if kind in ("mission", "lomo"):
        hm = set(held.mission_id)
        excl = set(evtr.event_key[evtr.mission_id.isin(hm)])
    else:
        excl = set(held.event_key)
    keys_train = [k for k in evtr.event_key if k not in excl]
    log(f"[{kind} fold {fi}] held {len(held)} eligible, missions {sorted(set(held.mission_id))}, train events {len(keys_train)}")
    fut = train_on(keys_train, seed=fi)
    torch.save(fut.state_dict(), os.path.join(OUT, f"mdn_{kind}_{fi}.pt"))
    # rollout targets: eligible held-out (evaluation) + for mission folds every train event of held-out missions (fans)
    if kind == "mission":
        tgt = evtr[evtr.event_key.isin(excl)]
    else:
        tgt = held
    r = roll(fut, tgt, NP, keep=(32 if kind == "mission" else 0), seed=1000 * fi)
    r["fold"] = np.full(len(tgt), fi); r["val_nll"] = np.full(len(tgt), fut.val_nll)
    allres.append(r)
    log(f"  rolled {len(tgt)} events x {NP} paths, mean steps {r['n_steps'].mean():.1f} ({time.time()-t0:.0f}s)")
    yc = (held.final_tier.values == 2)
    pk = dict(zip(r["event_key"], r["p_tiers"][:, 2]))
    pc = np.array([pk[k] for k in held.event_key])
    from sklearn.metrics import roc_auc_score
    log(f"  fold AUC {roc_auc_score(yc, pc) if 0 < yc.sum() < len(yc) else float('nan'):.3f}  mean p {pc.mean():.3f} obs {yc.mean():.3f}")

res = {}
for k in ["p_tiers", "q_lm", "q_rk", "n_steps", "final_ttca_med", "event_key", "fold", "val_nll"]:
    res[k] = np.concatenate([r[k] for r in allres])
if kind == "mission":
    res["paths"] = {k: np.concatenate([r["paths"][k] for r in allres]) for k in ["T", "L", "R"]}
res["grid"] = GRID
joblib.dump(res, os.path.join(OUT, f"futures_{kind}.joblib"))
log("done", f"{time.time()-t0:.0f}s")
