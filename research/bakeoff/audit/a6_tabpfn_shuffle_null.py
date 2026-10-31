"""Audit 6: is the tabpfn shuffled-label AUC (0.66 on mission fold 3) label leakage or an input-structure artifact?
Fit on 2 more permutations, save p_critical, compare against unsupervised scores built only from X (no labels)."""
import sys, json
sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd
import harness as H
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; yc = (y == 2).astype(int)
F = H.load_features("A13").loc[keys]; X = F.values.astype(np.float32); X[~np.isfinite(X)] = np.nan
out = {}
for k, trn, tst in H.folds("mission"):
    if k != 3: continue
    Ps = []
    for s in (11, 12):
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda:0", n_estimators=8, random_state=s)
        m.fit(X[trn], np.random.default_rng(s).permutation(y)[trn]); P = m.predict_proba(X[tst])
        Ps.append(P[:, 2] / P.sum(1)); out[f"shuf_seed{s}_auc"] = float(roc_auc_score(yc[tst], Ps[-1]))
    out["spearman_between_shuffles"] = float(spearmanr(Ps[0], Ps[1])[0])
    Xi = np.nan_to_num(X, nan=0.0)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Xi[trn]); sc = -iso.score_samples(Xi[tst])
    out["isolation_forest_outlier_auc"] = float(roc_auc_score(yc[tst], sc))
    out["spearman_shuf_vs_outlier"] = float(spearmanr(Ps[0], sc)[0])
    for c in ["last_max_risk_estimate", "last_log_miss", "n_cdm"]:
        out[f"spearman_shuf_vs_{c}"] = float(spearmanr(Ps[0], F[c].values[tst], nan_policy="omit")[0])
print(json.dumps(out, indent=1)); json.dump(out, open("/root/orbital-dodge/research/bakeoff/audit/a6_out.json", "w"), indent=1)
