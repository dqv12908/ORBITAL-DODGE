"""Kelvins-task ML done properly: P(high risk) classifier with an inner-CV F2-optimal threshold + value = latest risk (LRP) or regressor.
Compared to LRP baseline (clipped naive forecast). Same outer folds as b_baselines.py."""
import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.metrics import fbeta_score
tr = clean(load("train")); inp, tgt = kelvins_frame(tr)
numcols = [c for c in tr.columns if c not in ("event_id","mission_id") and tr[c].dtype.kind in "if"]
sets = {"A13": (ALLOWED[1:], TREND_COLS_A13), "A13+risk": (ALLOWED[1:]+["risk"], TREND_COLS_A13+["risk"]),
        "ALL": (numcols, TREND_COLS_A13+["risk","mahalanobis_distance","c_sigma_t","c_sigma_n","t_sigma_t","t_sigma_n"])}
y = tgt.y_risk.values; hr = (y>=-6).astype(int)
FE = {k: event_features(inp, list(v[0]), list(v[1])).loc[tgt.index] for k,v in sets.items()}
lat = FE["ALL"]["last_risk"].values
def kel(p):
    yc=np.maximum(y,-6.001); pc=np.maximum(p,-6.001); m=y>=-6
    f2=fbeta_score(m, p>=-6, beta=2); mse=np.mean((yc[m]-pc[m])**2); return f2, mse, mse/f2
lrp = np.where(lat>=-6, lat, -6.001)
for split in ["event","mission"]:
    folds = list(KFold(5,shuffle=True,random_state=0).split(lat)) if split=="event" else list(GroupKFold(5).split(lat, groups=tgt.mission_id.values))
    f2,mse,L = kel(lrp); print(f"\n== {split} ==  LRP baseline: F2={f2:.3f} MSE_HR={mse:.3f} L={L:.3f}  (recall {np.mean(lrp[hr==1]>=-6):.3f}, alerts {(lrp>=-6).sum()})")
    for fs in ["A13","A13+risk","ALL"]:
        X = FE[fs].values; p = np.zeros(len(y)); thr_used=[]
        for trn,tst in folds:
            # inner CV for threshold
            inner = np.zeros(len(trn))
            for itr, ite in StratifiedKFold(3,shuffle=True,random_state=1).split(trn, hr[trn]):
                c = HGC(max_iter=200,learning_rate=0.05,min_samples_leaf=20,random_state=0).fit(X[trn][itr], hr[trn][itr]); inner[ite]=c.predict_proba(X[trn][ite])[:,1]
            ths = np.linspace(0.01,0.6,60); best = ths[np.argmax([fbeta_score(hr[trn], inner>=t, beta=2) for t in ths])]; thr_used.append(best)
            c = HGC(max_iter=200,learning_rate=0.05,min_samples_leaf=20,random_state=0).fit(X[trn], hr[trn]); p[tst]=c.predict_proba(X[tst])[:,1]
        flag = np.zeros(len(y),bool)
        for (trn,tst),t in zip(folds,thr_used): flag[tst] = p[tst]>=t
        base = lat if "risk" in fs or fs=="ALL" else np.full(len(y), -5.5)
        val = np.where(flag, np.maximum(np.where(np.isnan(base),-5.5,base), -5.99), -6.001)
        f2,mse,L = kel(val)
        print(f"  {fs:9s} classifier(thr~{np.mean(thr_used):.2f}) + value=max(latest risk,-5.99): F2={f2:.3f} MSE_HR={mse:.3f} L={L:.3f} recall={np.mean(flag[hr==1]):.3f} alerts={flag.sum()}")
