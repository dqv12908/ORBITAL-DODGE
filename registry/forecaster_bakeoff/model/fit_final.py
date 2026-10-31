"""Final served lgbm_all model (bake-off winner). Declared recipe (Audit outcome, LEADERBOARD.md):

The served model is EXACTLY the per-fold procedure that refs.py evaluated, applied once to all 8,293 eligible events:
  LightGBM multiclass on features_ALL (233 cols), lr 0.03, 31 leaves, min_child_samples 20, subsample 0.8 (freq 1),
  colsample 0.8, lambda 1.0, seed 0; train on an 80% mission-grouped subset (GroupShuffleSplit(1, 0.2, random_state=0)
  over mission_id), early stopping (patience 100, multi_logloss) on the held-out 20% of missions; serve the booster at
  best_iteration. NO refit on all events (a refit is a procedure the OOF never evaluated). The refit variant the audit
  proposed is fitted here as a diagnostic only and is not served.
Calibration: isotonic regression on p_critical fitted on all 8,293 by-mission OOF predictions of lgbm_all
(the population of the cross-fitted isotonic score.py evaluated). Saved as threshold arrays (isotonic.json).
usage: .venv/bin/python models/lgbm_all/final/fit_final.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
BK = "/root/orbital-dodge/research/bakeoff"; sys.path.insert(0, BK)
import numpy as np, pandas as pd, lightgbm as lgb, sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
import harness as H
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values
F = H.load_features("ALL").loc[keys]; cols = F.columns.tolist(); X = F.values
P = dict(objective="multiclass", learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=16, verbose=-1)
itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=ev.mission_id.values))
m = lgb.LGBMClassifier(n_estimators=3000, **P).fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])],
                                                    callbacks=[lgb.early_stopping(100, verbose=False)])
nit = int(m.best_iteration_)
m.booster_.save_model(os.path.join(OUT, "lgbm_all.txt"), num_iteration=nit)
fit_s = time.time() - t0
# isotonic calibrator for p_critical from the by-mission OOF
oof = pd.read_parquet(os.path.join(H.OOF, "lgbm_all_mission.parquet")).set_index("event_key").loc[keys]
ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.p_critical.values, ev.y_crit.values)
json.dump({"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist(), "fitted_on": "lgbm_all by-mission OOF p_critical, 8293 events",
           "apply": "np.interp(p_critical, x, y) (clipped at the ends)"}, open(os.path.join(OUT, "isotonic.json"), "w"))
lb = json.load(open(os.path.join(BK, "leaderboard.json")))["splits"]["mission"]["models"]["lgbm_all"]
cfg = {"model": "lgbm_all", "version": "1.0.0", "feature_set": "features_ALL", "feature_columns": cols, "n_features": len(cols),
       "params": P, "early_stopping": {"inner_split": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0) on mission_id over all 8293 events",
       "patience": 100, "metric": "multi_logloss", "best_iteration": nit, "n_train_events": int(len(itr)), "n_valid_events": int(len(iva)),
       "n_train_missions": int(ev.mission_id.values[itr].astype(int).__array__().size and len(np.unique(ev.mission_id.values[itr]))),
       "n_valid_missions": int(len(np.unique(ev.mission_id.values[iva])))}, "refit_on_all": False,
       "classes": ["safe", "elevated", "critical"],
       "alert_threshold_p_critical": lb["primary_threshold"],
       "alert_threshold_note": "raw p_critical threshold that met the 113 false-alarm budget on the by-mission OOF (577 alerts / 8293 events)",
       "fit_seconds": round(fit_s, 1),
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__}}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print("best iter", nit, "fit", round(fit_s, 1), "s; missions train/valid", cfg["early_stopping"]["n_train_missions"], cfg["early_stopping"]["n_valid_missions"])
# test features: two independent builds must agree
T1 = pd.read_parquet(os.path.join(BK, "models/tabpfn/final/test_features_ALL.parquet"))
T2 = pd.read_parquet(os.path.join(BK, "audit/final/test_features_ALL.parquet"))
assert list(T1.columns) == cols and T1.index.equals(T2.index)
np.testing.assert_allclose(T1[cols].values, T2[cols].values, rtol=1e-12, atol=0, equal_nan=True)
T1.to_parquet(os.path.join(OUT, "test_features_ALL.parquet"))
sys.path.insert(0, OUT); import predict as PR
te = PR.predict(T1)
te.to_parquet(os.path.join(OUT, "test_predictions.parquet"))
print("test", te.shape, "alerts at threshold", int((te.p_critical >= cfg["alert_threshold_p_critical"]).sum()),
      "mean p_crit test", round(te.p_critical.mean(), 4), "train OOF", round(oof.p_critical.mean(), 4))
# diagnostic only: audit's refit-on-all variant
mr = lgb.LGBMClassifier(n_estimators=nit, **P).fit(X, y)
pr = mr.predict_proba(T1[cols].values)[:, 2]
thr = cfg["alert_threshold_p_critical"]; a1 = te.p_critical.values >= thr; a2 = pr >= thr
diag = {"refit_variant": "same params, n_estimators=best_iteration, fitted on all 8293 events (audit/final recipe); NOT served",
        "pearson_p_critical_test": float(np.corrcoef(te.p_critical.values, pr)[0, 1]),
        "alerts_served": int(a1.sum()), "alerts_refit": int(a2.sum()), "alerts_both": int((a1 & a2).sum())}
json.dump(diag, open(os.path.join(OUT, "refit_diagnostic.json"), "w"), indent=1); print(diag)
print("total", round(time.time() - t0, 1), "s")
