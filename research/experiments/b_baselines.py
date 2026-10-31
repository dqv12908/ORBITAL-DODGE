import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score, brier_score_loss, roc_auc_score, average_precision_score

tr = clean(load("train"))
inp, tgt = kelvins_frame(tr)
numcols = [c for c in tr.columns if c not in ("event_id","mission_id") and tr[c].dtype.kind in "if"]
sets = {
 "A13":      (ALLOWED[1:], TREND_COLS_A13),
 "A13+risk": (ALLOWED[1:]+["risk"], TREND_COLS_A13+["risk"]),
 "ALL":      (numcols, TREND_COLS_A13+["risk","mahalanobis_distance","c_sigma_t","c_sigma_n","t_sigma_t","t_sigma_n",
                                        "c_position_covariance_det","t_position_covariance_det","c_obs_used","c_weighted_rms"]),
}
FE = {k: event_features(inp, list(v[0]), list(v[1])).loc[tgt.index] for k, v in sets.items()}
for k,v in FE.items(): print(k, "features:", v.shape[1])
y = tgt.y_risk.values; yt = tier(tgt.y_risk, tgt.y_miss); ycrit = (yt==2).astype(int)
latest = FE["ALL"]
lat_risk = latest["last_risk"].values; lat_tier = tier(lat_risk, latest["last_miss_distance"].values)

def kelvins(y, p):
    yc = np.maximum(y, -6.001); pc = np.maximum(p, -6.001)
    hr = y >= -6
    f2 = fbeta_score(hr, p >= -6, beta=2)
    mse = np.mean((yc[hr]-pc[hr])**2)
    return f2, mse, mse/f2 if f2>0 else np.inf

def tier_report(yt, pt):
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=[0,1,2], zero_division=0)
    fa = int(((pt==2)&(yt!=2)).sum()); miss = int(((pt!=2)&(yt==2)).sum())
    return dict(rec_safe=r[0], rec_elev=r[1], rec_crit=r[2], prec_safe=p[0], prec_elev=p[1], prec_crit=p[2],
                macroF1=f.mean(), crit_alerts=int((pt==2).sum()), false_alarms=fa, missed_crit=miss)

def ece(y, p, bins=10):
    e = 0; edges = np.linspace(0,1,bins+1)
    for i in range(bins):
        m = (p>=edges[i])&(p<edges[i+1]) if i<bins-1 else (p>=edges[i])
        if m.sum(): e += m.sum()/len(p)*abs(y[m].mean()-p[m].mean())
    return e

def reliability(y, p, edges=(0,.02,.05,.1,.2,.4,.6,.8,1.0001)):
    rows=[]
    for a,b in zip(edges[:-1],edges[1:]):
        m=(p>=a)&(p<b)
        if m.sum(): rows.append(f"[{a:.2f},{min(b,1):.2f}) n={m.sum()} pred={p[m].mean():.3f} obs={y[m].mean():.3f}")
    return rows

results = {}
for split_name in ["event", "mission"]:
    if split_name=="event": folds = list(KFold(5, shuffle=True, random_state=0).split(latest))
    else: folds = list(GroupKFold(5).split(latest, groups=tgt.mission_id.values))
    print(f"\n================ GROUPED BY {split_name.upper()} ================")
    # persistence-empirical P(crit | latest tier) in-fold
    p_pers = np.zeros(len(y))
    for trn, tst in folds:
        for k in range(3):
            m = lat_tier[trn]==k
            p_pers[tst[lat_tier[tst]==k]] = ycrit[trn][m].mean()
    out = {}
    f2, mse, L = kelvins(y, lat_risk)
    print(f"[1] risk persistence   Kelvins F2={f2:.3f} MSE_HR={mse:.3f} L=MSE/F2={L:.3f}")
    r = tier_report(yt, lat_tier); out["persistence"]=r
    print(f"[2] tier persistence   " + " ".join(f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}" for k,v in r.items()))
    print(f"    P(crit) persistence-empirical Brier={brier_score_loss(ycrit,p_pers):.4f} ECE={ece(ycrit,p_pers):.4f} AUC={roc_auc_score(ycrit,p_pers):.3f}")
    print(f"    climatology Brier={brier_score_loss(ycrit, np.full(len(y), ycrit.mean())):.4f}")
    for fs in ["A13","A13+risk","ALL"]:
        X = FE[fs].values
        preg = np.zeros(len(y)); pcls = np.zeros((len(y),3)); pcls_b = np.zeros((len(y),3))
        for trn, tst in folds:
            reg = HGR(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=0)
            reg.fit(X[trn], np.maximum(y[trn], -7.0)); preg[tst] = reg.predict(X[tst])
            clf = HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
            clf.fit(X[trn], yt[trn]); pcls[tst] = clf.predict_proba(X[tst])
            clfb = HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0, class_weight="balanced")
            clfb.fit(X[trn], yt[trn]); pcls_b[tst] = clfb.predict_proba(X[tst])
        f2, mse, L = kelvins(y, preg)
        print(f"\n[{'3' if fs!='ALL' else '4'}] HGB {fs}: Kelvins F2={f2:.3f} MSE_HR={mse:.3f} L={L:.3f}")
        for lab, P in [("unweighted", pcls), ("balanced", pcls_b)]:
            r = tier_report(yt, P.argmax(1)); out[f"{fs}-{lab}"] = r
            print(f"    tier argmax {lab:10s} " + " ".join(f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}" for k,v in r.items()))
        pc = pcls[:,2]
        print(f"    P(crit) Brier={brier_score_loss(ycrit,pc):.4f} ECE={ece(ycrit,pc):.4f} AUC={roc_auc_score(ycrit,pc):.3f} AP={average_precision_score(ycrit,pc):.3f}")
        for line in reliability(ycrit, pc): print("      ", line)
        for th in [0.05, 0.1, 0.2, 0.3, 0.5]:
            a = pc>=th
            print(f"    alert if P(crit)>={th:.2f}: alerts={a.sum()} crit_recall={(a&(ycrit==1)).sum()/ycrit.sum():.3f} precision={(a&(ycrit==1)).sum()/max(a.sum(),1):.3f} false_alarms={(a&(ycrit==0)).sum()}")
        if split_name=="event" and fs in ("A13","ALL"):
            np.save(CACHE+f"/oof_{fs}_pcrit.npy", pc); np.save(CACHE+f"/oof_{fs}_reg.npy", preg)
    if split_name=="mission":
        # per-mission critical recall for persistence vs ALL
        df = pd.DataFrame({"m":tgt.mission_id.values,"yc":ycrit,"pers":(lat_tier==2).astype(int)})
        print("\nper-mission n / n_crit:", df.groupby("m").agg(n=("yc","size"), crit=("yc","sum")).T.to_dict())
np.save(CACHE+"/event_index.npy", tgt.index.values)
