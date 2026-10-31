import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from scipy.stats import pearsonr, spearmanr
raw = load("train"); ph = pd.read_parquet(CACHE+"/pc_phys.parquet"); d = raw.loc[ph.index].copy()
lp5 = np.load(CACHE+"/lp5.npy"); hbr = ((d.t_span+d.c_span)/2).values
lp = lp5 + 2*np.log10(hbr/5); y = d.risk.values; e = lp-y; m = y>=-10; ma = y>-30
print(f"HBR=(t_span+c_span)/2 [no fitting]: pearson(risk>-30, floored)={pearsonr(np.maximum(lp,-30)[ma],y[ma])[0]:.4f} spearman={spearmanr(lp[ma],y[ma])[0]:.4f}")
print(f"  MAE(risk>=-10)={np.mean(np.abs(e[m])):.3f} median|err|={np.median(np.abs(e[m])):.3f} p90={np.quantile(np.abs(e[m]),.9):.3f} within 0.1 dex={np.mean(np.abs(e[m])<0.1):.3f} within 0.5={np.mean(np.abs(e[m])<0.5):.3f}")
print(f"  floor agreement (both <=-30 or both >-30): {np.mean((lp<=-30)==(y<=-30)):.4f}")
for lo,hi in [(-30,-20),(-20,-10),(-10,-6),(-6,-4),(-4,0)]:
    mm = ((y>=lo) if lo>-30 else (y>-30)) & (y<hi)
    print(f"  risk in [{lo},{hi}): n={mm.sum()} MAE={np.mean(np.abs(e[mm])):.3f} median|err|={np.median(np.abs(e[mm])):.3f} bias={np.median(e[mm]):+.3f}")
for thr in [-6,-4]:
    yb=y>=thr; pb=lp>=thr
    print(f"  threshold {thr}: TP {int((yb&pb).sum())} FN {int((yb&~pb).sum())} FP {int((~yb&pb).sum())} recall {(yb&pb).sum()/yb.sum():.3f} precision {(yb&pb).sum()/pb.sum():.3f}")
# tier agreement on brief tiers: physics risk vs dataset risk, same miss
tp = tier(lp, d.miss_distance.values); ty = tier(y, d.miss_distance.values)
print("  brief-tier agreement (rows):", round(np.mean(tp==ty),4), pd.crosstab(ty, tp).to_dict())
d["err"]=e
print("  MAE by mission (risk>=-10):", d[m].groupby("mission_id").err.apply(lambda s: round(float(np.mean(np.abs(s))),2)).to_dict())
print("  median err by mission:", d[m].groupby("mission_id").err.median().round(2).to_dict())
print("  MAE by c_object_type:", d[m].groupby("c_object_type").err.apply(lambda s: round(float(np.mean(np.abs(s))),2)).to_dict())
bad = d[m & (np.abs(e)>1)]
print(f"  |err|>1 dex: {len(bad)} rows ({len(bad)/m.sum()*100:.1f}%); missions 6+7 share of bad: {bad.mission_id.isin([6,7]).mean():.3f} (share of all {d[m].mission_id.isin([6,7]).mean():.3f})")
excl = m & ~d.mission_id.isin([6,7]).values
print(f"  excluding missions 6,7: MAE={np.mean(np.abs(e[excl])):.3f} median={np.median(np.abs(e[excl])):.3f} within0.1={np.mean(np.abs(e[excl])<0.1):.3f}")
lo = m & (d.miss_distance.values>20000)
print(f"  miss>20 km rows: MAE={np.mean(np.abs(e[lo])):.3f} (n={lo.sum()}); miss<=5 km: MAE={np.mean(np.abs(e[m&(d.miss_distance.values<=5000)])):.3f}")
