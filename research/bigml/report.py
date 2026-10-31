"""Assemble research/bigml/results.json + registry/{futures,conformal}/ metrics and thresholds from out/eval.json."""
import os, json, joblib, numpy as np
from bm import HERE, ROOT

OUT = os.path.join(HERE, "out")
E = json.load(open(os.path.join(OUT, "eval.json")))
pol = json.load(open(os.path.join(ROOT, "registry", "policy", "metrics.json"))) if os.path.exists(os.path.join(ROOT, "registry", "policy", "metrics.json")) else None
G = joblib.load(os.path.join(OUT, "gbm_oof.joblib"))
yc = (G["y_tier"] == 2).astype(int); lat_tier = G["lat_tier"]
results = dict(generated="2026-10-31", setup=dict(task="P(final CDM tier = CRITICAL) from CDMs with ttca >= 2 d (Kelvins-eligible train events)",
               n_events=E["n_events"], n_critical=E["n_crit"], labels="brief tiers on final CDM dataset risk + miss (doc 11)",
               splits=dict(event="KFold(5, shuffle, seed 0)", mission="GroupKFold(5) on mission_id", lomo="leave-one-mission-out (19 folds)"),
               ci="95% bootstrap (event-level for by-event; mission-cluster bootstrap for by-mission / LOMO)"),
               methods=dict(persistence="tier of latest >=2 d CDM; P(crit) = in-fold empirical rate per latest tier",
                            hgb_A13="sklearn HGB 300 it, A13 84 features (design §3.5)",
                            lgbm_A13="LightGBM, Optuna-tuned (60 trials, tuned on a separate by-event split seed 99), mean of 10 seeds",
                            lgbm_ALL="same, 233 features incl. risk + covariance history",
                            futures_mc="mixture-density transition model CDM->CDM (10 comps, 3x256 MLP, GPU) + Monte Carlo rollouts to TCA (4,000 paths by-event/LOMO, 10,000 by-mission); P(crit) = share of futures ending CRITICAL; tier = argmax",
                            futures_mc_iso="futures_mc with cross-fitted isotonic calibration per class",
                            ens_futures_lgbmA13="average of futures_mc_iso and lgbm_A13 probabilities"),
               splits=E["splits"], conformal=E["conformal"], policy=pol, best_lgbm_params=G.get("best_params"))
results["summary"] = dict(
    headline="By held-out mission, risk futures (MDN + 10k Monte Carlo paths) beat tier persistence on ranking and probability quality (AUC 0.982 vs 0.958, Brier 0.0187 vs 0.0223, 100 vs 113 false alarms at persistence's 0.820 critical recall) but do NOT beat Optuna-tuned LightGBM (AUC 0.984-0.986, Brier 0.0174-0.0177, 94 FA at matched recall); CIs overlap. Best point predictor: average of futures and LightGBM A13 (91 FA at matched recall, recall 0.822 at argmax).",
    generalisation="By-mission and LOMO metrics match by-event for every learned model; the pre-registered 'does not generalise' trigger is not met.",
    conformal="Plain split conformal on futures P(crit), calibrated on other missions: realised recall 0.949 [0.925,0.967] at target 0.95 with 613 FA (persistence needs 1,323 FA and reaches only 0.926); 0.907 at target 0.90 with 257 FA. Guarantee is marginal: mission 15 falls below target for every score. Mondrian-by-latest-tier fails (about 7,000 alerts) - negative result.",
    policy="Value-iteration policy on 588 discretised CDM states with real + MDN-simulated transitions: mean loss 0.859/event [0.811,0.913] on held-out missions vs 1.039 for 'prepare if critical', 3.896 for the brief rule (tracking at every ELEVATED CDM), oracle 0.332; 0 missed criticals. TRACK is almost never optimal under the stated costs.",
    failure_cases=["train:9361 critical by Pc 1.2e-4 at 3.93 km: futures P(crit)=0.04 (Pc-driven criticals are 6/551)",
                   "mission 15 late escalations: LOMO crit recall 0.685 (persistence and futures), 0.704 (LightGBM); train:2563 SAFE at T-2.43 d gets futures P(crit)=0.092 >= conformal tau95 0.067, so it would be flagged 2 d early",
                   "mission 6: crit recall 0.50 for every method (8 criticals)",
                   "fans are on the CDM risk scale (floor -30), not physics Pc; raw MC P(crit) slightly over-confident (ECE 0.008)"])
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)

# registry/futures/metrics.json
fm = {sp: {m: {k: v for k, v in E["splits"][sp][m].items() if k != "per_mission"} for m in E["splits"][sp]
           if m in ("futures_mc", "futures_mc_iso", "persistence", "hgb_A13", "lgbm_A13")} for sp in E["splits"]}
json.dump(dict(model_version="futures-mdn-1.0", task=results["setup"]["task"], metrics=fm), open(os.path.join(ROOT, "registry", "futures", "metrics.json"), "w"), indent=1)

# registry/conformal: thresholds fitted on ALL by-mission OOF scores (for use with the full models' scores)
os.makedirs(os.path.join(ROOT, "registry", "conformal"), exist_ok=True)
F = joblib.load(os.path.join(OUT, "futures_mission.joblib"))
import pandas as pd
idx = pd.Series(np.arange(len(F["event_key"])), index=F["event_key"])
keys = ["train:%d" % i for i in G["ids"]]
scores = {"futures_mc": F["p_tiers"][idx.loc[keys].values, 2], "lgbm_A13": G["oof"][("lgbm_A13", "mission")][:, 2],
          "hgb_A13": G["oof"][("hgb_A13", "mission")][:, 2]}
th = {}
for name, s in scores.items():
    th[name] = {}
    for alpha in [0.05, 0.10]:
        cal = np.sort(s[yc == 1]); k = int(np.floor(alpha * (len(cal) + 1)))
        plain = float(cal[k - 1])
        mond = {}
        for t, tn in enumerate(["SAFE", "ELEVATED", "CRITICAL"]):
            c = np.sort(s[(yc == 1) & (lat_tier == t)]); kk = int(np.floor(alpha * (len(c) + 1)))
            mond[tn] = float(c[kk - 1]) if kk >= 1 and len(c) else 0.0
        th[name][f"recall_{1-alpha:.2f}"] = dict(tau=plain, mondrian_by_latest_tier=mond, n_cal_critical=int(len(cal)))
json.dump(dict(model_version="conformal-1.0",
               rule="alert if p_critical >= tau (class-conditional split conformal on critical events; guarantee: P(alert | final CRITICAL) >= target, marginal over exchangeable events)",
               calibration="by-mission out-of-fold scores of the 551 final-critical eligible train events",
               thresholds=th), open(os.path.join(ROOT, "registry", "conformal", "thresholds.json"), "w"), indent=1)
json.dump(dict(model_version="conformal-1.0", evaluation=E["conformal"]), open(os.path.join(ROOT, "registry", "conformal", "metrics.json"), "w"), indent=1)
print("written results.json, registry futures/conformal metrics + thresholds")
