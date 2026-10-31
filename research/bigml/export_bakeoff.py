"""Export MC risk-futures OOF predictions to research/bakeoff/oof/futures_{event,mission}.parquet.
Checks our folds against research/bakeoff/folds.parquet when present."""
import os, joblib, numpy as np, pandas as pd
from bm import HERE, ROOT, load_rows, events_table, folds

OUT = os.path.join(HERE, "out")
BK = os.path.join(ROOT, "research", "bakeoff")
os.makedirs(os.path.join(BK, "oof"), exist_ok=True)
d = load_rows(); ev = events_table(d)
el = ev[(ev.split == "train") & ev.eligible].sort_values("event_id").reset_index(drop=True)
mine = {}
for kind in ["event", "mission"]:
    f = np.zeros(len(el), int)
    for i, (a, b) in enumerate(folds(el, kind)):
        f[b] = i
    mine[kind] = f
fp = os.path.join(BK, "folds.parquet")
if os.path.exists(fp):
    ref = pd.read_parquet(fp).set_index("event_key").loc[el.event_key]
    for kind in ["event", "mission"]:
        same = (ref[f"fold_{kind}"].values == mine[kind]).all()
        print(f"folds.parquet {kind}: identical to ours = {same}")
else:
    print("no folds.parquet; using KFold(5,shuffle,0) / GroupKFold(5) over eligible events sorted by event_id")
for kind in ["event", "mission"]:
    p = os.path.join(OUT, f"futures_{kind}.joblib")
    if not os.path.exists(p):
        print("missing", p); continue
    F = joblib.load(p)
    npaths = 10000 if kind == "mission" else 4000
    df = pd.DataFrame(dict(event_key=F["event_key"], p_safe=F["p_tiers"][:, 0], p_elevated=F["p_tiers"][:, 1],
                           p_critical=F["p_tiers"][:, 2]))
    df = df[df.event_key.isin(el.event_key)].set_index("event_key").loc[el.event_key].reset_index()
    # Wilson 95% interval for the Monte Carlo share (MC sampling error only, not model error)
    z = 1.96; ph = df.p_critical.values; nn = npaths
    c = (ph + z * z / (2 * nn)) / (1 + z * z / nn); h = z * np.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
    df["p_critical_lo"] = np.clip(c - h, 0, 1); df["p_critical_hi"] = np.clip(c + h, 0, 1)
    df["fold"] = mine[kind]; df["n_paths"] = npaths; df["model_version"] = "futures-mdn-1.0"
    df.to_parquet(os.path.join(BK, "oof", f"futures_{kind}.parquet"), index=False)
    print("wrote", kind, len(df))
