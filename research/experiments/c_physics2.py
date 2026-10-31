import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
raw = load("train"); ph = pd.read_parquet(CACHE+"/pc_phys.parquet")
df = raw.loc[ph.index].copy(); df["pc_phys"]=ph.pc_phys; df["err"]=ph.err
m = df.risk>=-10
d = df[m]
print("implied HBR per mission (median offset, rotate, HBR*=3.71 baseline):")
g = d.groupby("mission_id").apply(lambda x: pd.Series({"n":len(x),"median_err":np.median(x.err),"MAE":np.mean(np.abs(x.err)),
      "implied_HBR_m": 3.71*10**(-np.median(x.err)/2)}))
print(g.round(2).to_string())
print("\nerror by c_object_type:"); print(d.groupby("c_object_type").err.agg(["size","median",lambda s: np.mean(np.abs(s))]).round(3).to_string())
d2 = d.assign(ttbin=pd.cut(d.time_to_tca,[-1,1,2,3,4,5,7]))
print("\nerror by time_to_tca bin:"); print(d2.groupby("ttbin").err.agg(["size","median",lambda s: np.mean(np.abs(s))]).round(3).to_string())
d3 = d.assign(sbin=pd.qcut(d.c_sigma_t,5))
print("\nerror by c_sigma_t quintile:"); print(d3.groupby("sbin").err.agg(["size","median",lambda s: np.mean(np.abs(s))]).round(3).to_string())
d4 = d.assign(mb=pd.qcut(d.max_risk_scaling,5))
print("\nerror by max_risk_scaling quintile:"); print(d4.groupby("mb").err.agg(["size","median",lambda s: np.mean(np.abs(s))]).round(3).to_string())
# risk vs max_risk_estimate relationship
print("\nrisk <= max_risk_estimate fraction:", round((raw.risk<=raw.max_risk_estimate+1e-6).mean(),4))
# learned crude: row-level HGB on 13 allowed cols predicting risk (grouped by event)
X = clean(raw)[ALLOWED[1:]].copy(); X["log_miss"]=np.log10(raw.miss_distance.clip(1))
y = raw.risk.values; oof = np.zeros(len(y))
for trn,tst in GroupKFold(5).split(X, groups=raw.event_id):
    r = HGR(max_iter=300, learning_rate=0.1, random_state=0).fit(X.iloc[trn], y[trn]); oof[tst]=r.predict(X.iloc[tst])
mm = y>=-10; ma = y>-30
print(f"\nLEARNED crude (HGB, 13 cols incl max_risk_estimate, row-level, GroupKFold event): MAE(risk>=-10)={np.mean(np.abs(oof[mm]-y[mm])):.3f} pearson(risk>-30)={pearsonr(oof[ma],y[ma])[0]:.3f} spearman={spearmanr(oof[ma],y[ma])[0]:.3f}")
yb=y>=-6; pb=oof>=-6; print("   HR(-6) TP",int((yb&pb).sum()),"FN",int((yb&~pb).sum()),"FP",int((~yb&pb).sum()))
# physics + learned on the same subset
idx = raw.index.get_indexer(ph.index); sub = np.zeros(len(raw),bool); sub[idx]=True
mm2 = mm & sub
print(f"   on physics-subset rows risk>=-10: learned MAE={np.mean(np.abs(oof[mm2]-y[mm2])):.3f} vs physics MAE={np.mean(np.abs(ph.err.values[(ph.risk>=-10).values])):.3f}")
# hybrid: physics pc as feature
X2 = X.copy(); X2["pc_phys"]=np.nan; X2.loc[ph.index,"pc_phys"]=ph.pc_phys.clip(-60)
oof2=np.zeros(len(y))
for trn,tst in GroupKFold(5).split(X2, groups=raw.event_id):
    r = HGR(max_iter=300, learning_rate=0.1, random_state=0).fit(X2.iloc[trn], y[trn]); oof2[tst]=r.predict(X2.iloc[tst])
print(f"   hybrid (13 cols + physics Pc): MAE(risk>=-10)={np.mean(np.abs(oof2[mm]-y[mm])):.3f} spearman={spearmanr(oof2[ma],y[ma])[0]:.3f}")
