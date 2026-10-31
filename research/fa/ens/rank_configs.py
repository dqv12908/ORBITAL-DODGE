import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, "/root/orbital-dodge/research/sweep")
import nested as N, cost as C
recs=[]
for f in glob.glob("/root/orbital-dodge/research/sweep/configs/*.jsonl"):
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if r.get("status")=="ok": recs.append(r)
print(len(recs))
lab=None
out=[]
for r in recs:
    try: d=pd.read_parquet(r["path"])
    except Exception as e: continue
    if lab is None:
        lab=N._events().set_index("event_key")["y_crit"]
    io=d[d.role=="inner_oof"]
    cs=[]
    for k,g in io.groupby("outer_fold"):
        y=lab.loc[g.event_key.values].values
        t=C.empirical_threshold(g.p_critical.values,y,10,recall_floor=0.9)
        cs.append(t["cost"]/len(g)*1000)
    out.append((r["config_id"],r["family"],r["model"],r["features"],np.mean(cs)))
df=pd.DataFrame(out,columns=["cid","fam","model","feat","inner_cost"]).sort_values("inner_cost")
df.to_csv("/root/orbital-dodge/research/fa/ens/config_rank.csv",index=False)
pd.set_option("display.width",200); print(df.head(40).to_string())
