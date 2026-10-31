"""Export the served bake-off forecaster to /root/orbital-dodge/registry/forecaster_bakeoff/ and find the shared failure case.
usage: .venv/bin/python export_registry.py"""
import sys, os, json, shutil, warnings, datetime
warnings.filterwarnings("ignore")
BK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BK)
import numpy as np, pandas as pd, lightgbm as lgb, sklearn
import harness as H, score as S
REG = "/root/orbital-dodge/registry/forecaster_bakeoff"; FIN = os.path.join(BK, "models/lgbm_all/final")
os.makedirs(os.path.join(REG, "model"), exist_ok=True)
lbj = json.load(open(os.path.join(BK, "leaderboard.json"))); cfg = json.load(open(os.path.join(FIN, "config.json")))
diag = json.load(open(os.path.join(FIN, "refit_diagnostic.json")))
ev = H.load_events(); keys = ev.event_key.values; yc = ev.y_crit.values
W = lbj["decision"]["winner"]; assert W == "lgbm_all" and lbj["decision"]["serve"]
# ---- predictions: train OOF by-mission (cross-fitted isotonic, as score.py) + all test events (final model + full isotonic)
oof = pd.read_parquet(os.path.join(H.OOF, f"{W}_mission.parquet")).set_index("event_key").loc[keys]
fold = H.fold_ids("mission")
tr = pd.DataFrame({"event_key": keys, "p_safe": oof.p_safe.values, "p_elevated": oof.p_elevated.values, "p_critical": oof.p_critical.values})
tr["p_critical_calibrated"] = S.isotonic_crossfit(tr.p_critical.values, yc, fold)
tr["model"] = W; tr["split"] = "train_oof_mission"
te = pd.read_parquet(os.path.join(FIN, "test_predictions.parquet")).drop(columns=["alert"])
te["model"] = W; te["split"] = "test"
pred = pd.concat([tr, te], ignore_index=True)
assert pred.event_key.is_unique and len(tr) == 8293 and len(te) == 2167
assert np.allclose(pred[H.PCOLS].sum(1), 1, atol=1e-6) and np.isfinite(pred[H.PCOLS + ["p_critical_calibrated"]].values).all()
pred.to_parquet(os.path.join(REG, "predictions.parquet"), index=False)
for f in ["lgbm_all.txt", "config.json", "isotonic.json", "predict.py", "fit_final.py", "refit_diagnostic.json"]:
    shutil.copy(os.path.join(FIN, f), os.path.join(REG, "model", f))
# ---- failure case: critical events no row (incl. persistence) alerts on at its own by-mission budget threshold
models = list(lbj["splits"]["mission"]["models"])
caught = np.zeros(len(keys), bool)
for mname in models:
    p = S.load_oof(mname, "mission", keys)[:, 2]
    caught |= p >= lbj["splits"]["mission"]["models"][mname]["primary_threshold"]
missed = np.where((yc == 1) & ~caught)[0]
raw = pd.read_parquet(os.path.join(H.DATA, "raw_train.parquet"))
cand = ev.iloc[missed][["event_key", "event_id", "mission_id", "latest_tier", "latest_ttca", "n_cdm_ge2d"]].copy()
cand["p_crit_lgbm_all"] = oof.p_critical.values[missed]
print("critical events missed by every model:", len(missed)); print(cand.sort_values("n_cdm_ge2d", ascending=False).head(10).to_string())
# pick: most >=2 d CDMs (richest history), tie -> lowest event_id
pick = cand.sort_values(["n_cdm_ge2d", "event_id"], ascending=[False, True]).iloc[0]
traj = raw[raw.event_id == pick.event_id].sort_values("time_to_tca", ascending=False)[["time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed", "c_object_type", "t_sigma_r", "c_sigma_r"]]
traj["tier"] = np.array(H.TIERS)[H.tier(traj.risk, traj.miss_distance)]
traj["input_to_model"] = traj.time_to_tca >= 2
pm = {m: float(S.load_oof(m, "mission", keys)[ev.index[ev.event_key == pick.event_key][0], 2]) for m in models}
fail = {"event_key": pick.event_key, "mission_id": int(pick.mission_id), "n_missed_by_all": int(len(missed)),
        "p_critical_by_model_mission_oof": pm, "trajectory": traj.round(4).to_dict(orient="records")}
json.dump(fail, open(os.path.join(BK, "failure_case.json"), "w"), indent=1, default=str)
print(traj.round(3).to_string()); print(json.dumps(pm, indent=0))
# ---- model card
mm = lambda sp, m: lbj["splits"][sp]["models"][m]
def block(sp, m):
    r = mm(sp, m)
    return {k: r[k] for k in ["primary", "primary_ci", "primary_threshold", "primary_false_alarms", "auc", "auc_ci", "ap", "brier", "brier_ci",
                              "ece", "ece_iso", "brier_iso", "macro_f1", "at_0_50", "at_0_10", "delta_primary_vs_persistence", "delta_primary_ci", "p_delta_le_0"]}
card = {
 "name": "forecaster_bakeoff", "served_model": W, "version": "1.0.0", "created": datetime.date.today().isoformat(),
 "task": "predict the brief tier (safe/elevated/critical) of the FINAL CDM (<1 d before TCA) from the CDMs available at the 2-day decision point (time_to_tca >= 2 d)",
 "label": "critical if final risk >= -4 or miss < 1000 m; elevated if risk >= -6 or miss <= 5000 m; else safe. 6138 / 1604 / 551 over 8293 eligible train events",
 "model_family": "LightGBM multiclass gradient-boosted trees (lightgbm.LGBMClassifier, objective multiclass)",
 "features": {"set": "features_ALL", "n": cfg["n_features"], "built_by": "research/bakeoff/harness.py event_features over CDMs with time_to_tca >= 2 d only",
              "description": "latest-CDM values of all numeric columns + derived log_miss, sig_r_comb, r_over_sig, cov_unknown; n_cdm, first_ttca; d1/dall/min/max/std/slope trends of 21 columns (miss, speed, rel. position, sigmas, risk, max_risk_estimate, mahalanobis, covariance dets, c_obs_used, c_weighted_rms)",
              "columns": cfg["feature_columns"], "leakage_guard": "no final CDM, no CDM < 2 d, no final risk/miss/ttca in any feature"},
 "hyperparameters": cfg["params"], "final_fit": cfg["early_stopping"] | {"refit_on_all": False},
 "final_fit_refit_diagnostic": diag,
 "calibration": {"p_critical_calibrated_train": "cross-fitted isotonic over the 5 by-mission folds (identical to score.py)",
                 "p_critical_calibrated_test": "isotonic fitted on all 8293 by-mission OOF p_critical (model/isotonic.json)",
                 "note": "only p_critical is calibrated; p_safe/p_elevated are raw and the calibrated value is not renormalised"},
 "alert_threshold_p_critical": cfg["alert_threshold_p_critical"], "alert_threshold_note": cfg["alert_threshold_note"],
 "folds": {"decision": "by-mission: GroupKFold(5) on mission_id (data/folds.parquet fold_mission)", "sanity": "by-event: KFold(5, shuffle, random_state=0)",
           "fold_sizes_mission": np.bincount(fold).tolist(), "inner": "early stopping on a 20% mission-grouped inner split of the training folds"},
 "evaluation": {"bootstrap": {"B": lbj["B"], "seed": lbj["boot_seed"], "paired": True, "unit": "event"},
                "n_events": lbj["n_events"], "n_critical": lbj["n_critical"], "fa_budget": {sp: lbj["splits"][sp]["fa_budget"] for sp in ("mission", "event")},
                "by_mission": block("mission", W), "by_event": block("event", W)},
 "persistence_comparison": {"by_mission": {"persistence": block("mission", "persistence"), "delta_primary": mm("mission", W)["delta_primary_vs_persistence"], "delta_primary_ci": mm("mission", W)["delta_primary_ci"]},
                            "by_event": {"persistence": block("event", "persistence"), "delta_primary": mm("event", W)["delta_primary_vs_persistence"], "delta_primary_ci": mm("event", W)["delta_primary_ci"]}},
 "decision_rule": {"source": "research/bakeoff/PREREG.md section 5 (sha256 in PREREG.sha256, no amendments)", **lbj["decision"],
                   "audit_outcome": "repro finding (no final model / undeclared final-fit recipe) fixed: recipe declared and model built; OOF scores unchanged, rule re-applied, same verdict. See LEADERBOARD.md 'Audit outcome'."},
 "known_limits": [
   "Gate passed via (b) non-inferiority + AUC, not via (a): paired dPRIMARY CI is [+0.000, +0.041]; the lead over persistence is 12 of 551 critical events (464 vs 452).",
   "The by-mission lead does not reproduce by-event (PRIMARY 0.820 = persistence). Runner-up tabpfn (0.838, gate (a)) is 2 events behind, well inside the CI.",
   "Seed sensitivity (audit, LightGBM seeds 0-4 and 7, by-mission): PRIMARY 0.828-0.842; seed 0 is the pre-registered seed and the top of that range, so expect regression toward ~0.835.",
   "Only 19 missions: by-mission folds hold out 3-4 missions each, and the final model's early stopping used 4 held-out missions (161 iterations; OOF folds stopped at 101-131).",
   "Served model is trained on 80% of events (the evaluated procedure); the refit-on-all variant agrees closely on test (see final_fit_refit_diagnostic) but is not served.",
   "Test set (2,167 events) has no labels here (Kelvins Zenodo release deliberately not used), so test predictions are unscored. Mean test p_critical 0.079 vs 0.064 on train OOF; 191 / 2167 test events alert at the threshold (8.8%) vs 577 / 8293 (7.0%) on train.",
   "Tier label depends on the final CDM which can jump late (<2 d); about 16% of critical events are missed at the persistence false-alarm budget, and some are missed by every model (failure_case in docs/BAKEOFF.md).",
   "p_critical_calibrated is calibrated only for the critical class; ECE after isotonic 0.005 (by-mission)."],
 "runtime": {"oof": lbj["runtime"][W], "final_fit_seconds": cfg["fit_seconds"], "server": "shared 56-core / 4-GPU server under heavy load; CPU only (n_jobs=16)"},
 "library_versions": cfg["versions"] | {"pyarrow": __import__("pyarrow").__version__},
 "files": {"predictions.parquet": "event_key, p_safe, p_elevated, p_critical, p_critical_calibrated, model, split (train_oof_mission: 8293, test: 2167)",
           "model/": "lgbm_all.txt booster, config.json, isotonic.json, predict.py, fit_final.py, refit_diagnostic.json",
           "leaderboard.json": "full bake-off scores (score.py output) + audit outcome"},
}
json.dump(card, open(os.path.join(REG, "model_card.json"), "w"), indent=1)
lbo = dict(lbj); lbo["audit_outcome"] = card["decision_rule"]["audit_outcome"]; lbo["served"] = {"model": W, "version": "1.0.0"}
json.dump(lbo, open(os.path.join(REG, "leaderboard.json"), "w"), indent=1)
print("exported", os.listdir(REG), pred.groupby("split").size().to_dict())
