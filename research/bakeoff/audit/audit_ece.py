import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
B="/root/orbital-dodge/research/bakeoff"
ev=pd.read_parquet(f"{B}/data/events.parquet"); yc=(ev.y_tier.values==2).astype(float); keys=ev.event_key.values
f=pd.read_parquet(f"{B}/folds.parquet").set_index("event_key").loc[keys].fold_mission.values
def ece(y,p):
    b=np.minimum((p*10).astype(int),9); return sum((b==k).mean()*abs(y[b==k].mean()-p[b==k].mean()) for k in range(10) if (b==k).any())
for m in ["lgbm_all","tabpfn","persistence"]:
    p=pd.read_parquet(f"{B}/oof/{m}_mission.parquet").set_index("event_key").loc[keys].p_critical.values
    q=np.zeros_like(p)
    for k in np.unique(f):
        ir=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(p[f!=k],yc[f!=k]); q[f==k]=ir.predict(p[f==k])
    print(m,"ECE",round(ece(yc,p),5),"ECE_iso",round(ece(yc,q),5))
