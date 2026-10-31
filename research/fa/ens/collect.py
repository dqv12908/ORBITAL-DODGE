import glob, joblib, pandas as pd, sys
sys.argv=[sys.argv[0]]
exec(open("run.py").read().split("jobs = [")[0])
for m in FEAT:
    fs = glob.glob(f"{OUTD}/t_{m}_*.joblib")
    if len(fs) < 60: print("incomplete", m, len(fs)); continue
    rows = []
    for f in fs:
        mm, oi, j, ix, p = joblib.load(f); sp, s, kind, trn, tst = outer[oi]
        rows.append(pd.DataFrame({"idx": ix, "split": sp, "fold": s if sp == "mission" else 0, "inner_fold": j,
                                  "role": "outer_test" if j < 0 else "inner_oof", "p_critical": p}))
    pd.concat(rows).to_parquet(f"{OUTD}/{m}.parquet")
