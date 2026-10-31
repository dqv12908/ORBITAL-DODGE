"""FA reference: export current LGBM-ALL (organizer params, seed 0), persistence, and lgbm_all_bag (10 seeds, bagging)
in the standard candidate format.  usage: ref.py ref|bag  [NJ]"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
ORGD = "/root/orbital-dodge/research/organizer"
sys.path.insert(0, ORGD); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold
import organizer as O
CANDS = "/root/orbital-dodge/research/fa/cands"; os.makedirs(CANDS, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
mode = sys.argv[1]; NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ev, FE = O.load(); X = FE["ALL"]; y = ev.y_tier.values; lt = ev.latest_tier.values; n = len(ev)
BEST = O.BEST["ALL"]
# c_object_type of the latest input message (ttca >= 2 d)
raw = pd.read_parquet("/root/orbital-dodge/research/bakeoff/data/raw_train.parquet", columns=["event_id", "time_to_tca", "c_object_type"])
raw = raw[raw.time_to_tca >= 2].sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
cot = raw.groupby("event_id").tail(1).set_index("event_id").c_object_type.reindex(ev.event_id.values).values

outer = []
for s in range(5):
    trn, tst = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(np.arange(n), groups=ev.event_id.values))
    outer.append((f"8020_s{s}", 0, s, "event80_20", np.sort(trn), np.sort(tst)))
for k, (trn, tst) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)):
    outer.append(("mission", k, k, "mission", trn, tst))

def fit(params, a, b, seed):
    p = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=1, seed=seed, **params)
    nr = p.pop("n_estimators")
    return lgb.train(p, lgb.Dataset(X[a], y[a]), num_boost_round=nr).predict(X[b])

def frame(split, fold, role, idx, P):
    return pd.DataFrame({"event_key": ev.event_id.values[idx], "split": split, "fold": fold, "role": role,
                         "p_critical": P[:, 2], "p_safe": P[:, 0], "p_elevated": P[:, 1], "y_tier": y[idx],
                         "latest_tier": lt[idx], "c_object_type": cot[idx], "final_sub": ev.subpop.values[idx]})

def run(name, params, seeds):
    jobs = []  # (outer idx, part (-1 = outer test, else inner fold), a, b, seed)
    for oi, (sp, fold, s, osp, trn, tst) in enumerate(outer):
        for sd in seeds: jobs.append((oi, -1, trn, tst, sd))
        for j, (ia, ib) in enumerate(O.inner_folds(osp, trn, ev, s)):
            for sd in seeds: jobs.append((oi, j, trn[ia], trn[ib], sd))
    log(name, "fits", len(jobs))
    outs = Parallel(n_jobs=NJ)(delayed(fit)(params, a, b, sd) for (_, _, a, b, sd) in jobs)
    acc = {}
    for (oi, part, a, b, sd), P in zip(jobs, outs):
        key = (oi, part)
        if key not in acc: acc[key] = [b, np.zeros((len(b), 3))]
        acc[key][1] += P / len(seeds)
    dfs = []
    for oi, (sp, fold, s, osp, trn, tst) in enumerate(outer):
        b, P = acc[(oi, -1)]; dfs.append(frame(sp, fold, "outer_test", b, P))
        inn = [acc[k] for k in acc if k[0] == oi and k[1] >= 0]
        bi = np.concatenate([u[0] for u in inn]); Pi = np.concatenate([u[1] for u in inn])
        o = np.argsort(bi); dfs.append(frame(sp, fold, "inner_oof", bi[o], Pi[o]))
    df = pd.concat(dfs, ignore_index=True); df.to_parquet(os.path.join(CANDS, f"{name}.parquet")); log("wrote", name, df.shape)
    return df

def persistence():
    dfs = []
    for sp, fold, s, osp, trn, tst in outer:
        dfs.append(frame(sp, fold, "outer_test", tst, O.persistence_P(lt, y, trn, tst)))
        Pi = np.zeros((len(trn), 3))
        for ia, ib in O.inner_folds(osp, trn, ev, s):
            Pi[ib] = O.persistence_P(lt, y, trn[ia], trn[ib])
        dfs.append(frame(sp, fold, "inner_oof", trn, Pi))
    df = pd.concat(dfs, ignore_index=True); df.to_parquet(os.path.join(CANDS, "persistence.parquet")); log("wrote persistence", df.shape)
    return df

def quick(df, name):
    res = {}
    for grp, sps in (("8020", [f"8020_s{s}" for s in range(5)]), ("mission", ["mission"])):
        rows = []; tot = dict(TP=0, FA=0, FN=0, n=0)
        for sp in sps:
            for f in sorted(df[df.split == sp].fold.unique()):
                d = df[(df.split == sp) & (df.fold == f)]
                ii = d[d.role == "inner_oof"]; oo = d[d.role == "outer_test"]
                t, _ = O.pick_thr((ii.y_tier.values == 2).astype(int), ii.p_critical.values)
                al = oo.p_critical.values >= t; yc = oo.y_tier.values == 2
                r = dict(TP=int((al & yc).sum()), FA=int((al & ~yc).sum()), FN=int((~al & yc).sum()), n=len(oo))
                if grp == "8020": rows.append(r)
                else:
                    for k in tot: tot[k] += r[k]
        if grp == "mission": rows = [tot]
        R = pd.DataFrame(rows)
        res[grp] = dict(recall=round(float((R.TP / (R.TP + R.FN)).mean()), 4), FA=round(float(R.FA.mean()), 1),
                        cost_per_1000=round(float(((10 * R.FN + R.FA) / R.n * 1000).mean()), 1))
    log("QUICK", name, json.dumps(res))
    json.dump(res, open(f"/root/orbital-dodge/research/fa/ref/quick_{name}.json", "w"))
    return res

if mode == "ref":
    quick(persistence(), "persistence")
    quick(run("current_lgbm_all", BEST, [0]), "current_lgbm_all")
elif mode == "bag":
    pb = dict(BEST); pb["bagging_fraction"] = 0.8; pb["bagging_freq"] = 1
    quick(run("lgbm_all_bag", pb, list(range(10))), "lgbm_all_bag")
