import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from scipy.stats import spearmanr
raw = load("train"); ph = pd.read_parquet(CACHE+"/pc_phys.parquet")
d = raw.loc[ph.index]
print("t_span by mission (median):", d.groupby("mission_id").t_span.median().round(2).to_dict())
print("c_span describe:", d.c_span.describe().round(2).to_dict())
print("c_span by object type (median):", d.groupby("c_object_type").c_span.median().round(2).to_dict())
# the pc_phys in cache uses per-mission HBR; reconstruct a 5 m version: pc5 = pc_phys - 2log10(hbr_m/5); simpler: recompute implied HBR per row
from pc2d import pc2d_log10
lp5 = pc2d_log10(d, 5.0, rotate=True)[0]
y = d.risk.values; m = y >= -10
for name, h in [("t_span/2", d.t_span/2), ("t_span", d.t_span), ("(t_span+c_span)/2", (d.t_span+d.c_span)/2), ("t_span/2 + c_span/2 (same)", d.t_span/2+d.c_span/2),("t_span+c_span", d.t_span+d.c_span)]:
    lp = lp5 + 2*np.log10(h.values/5); e = lp-y
    print(f"HBR={name:28s} MAE(risk>=-10)={np.mean(np.abs(e[m])):.3f} median|err|={np.median(np.abs(e[m])):.3f} within0.1={np.mean(np.abs(e[m])<0.1):.3f} bias={np.median(e[m]):+.3f}  risk>=-4 MAE={np.mean(np.abs(e[y>=-4])):.3f}")
np.save(CACHE+"/lp5.npy", lp5)
