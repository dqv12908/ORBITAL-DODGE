"""Assemble research/organizer/results.json (+ summary narrative) and registry/forecaster_organizer/1.0.0/model_card.json."""
import os, json, shutil, datetime
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
REG = os.path.join(HERE, "registry", "forecaster_organizer", "1.0.0"); os.makedirs(REG, exist_ok=True)
b = json.load(open(os.path.join(HERE, "data", "build_summary.json")))
m = json.load(open(os.path.join(OUT, "eval_metrics.json")))
f = json.load(open(os.path.join(OUT, "final_summary.json")))
REC = "lgbm_all__train_organizer"


def p(sp, mod, grp="all"):
    return m[sp][mod][grp]["point"]


def ci(sp, mod, k, grp="all"):
    lo, hi = m[sp][mod][grp]["ci95"][k]; return f"[{lo:.3g}, {hi:.3g}]"


def line(sp, mod, grp="all"):
    q = p(sp, mod, grp)
    return (f"{mod} [{sp}, {grp}]: crit recall {q['recall']:.3f} {ci(sp, mod, 'recall', grp)}, FA {q['false_alarms']:.1f}, "
            f"cost/1000 ev {q['cost_per_1000']:.1f} {ci(sp, mod, 'cost_per_1000', grp)}, macro-F1 {q['macro_f1']:.3f} "
            f"{ci(sp, mod, 'macro_f1', grp)}, AUC {q['auc_crit']:.4f}, tier recall S/E/C {q['recall_safe']:.3f}/"
            f"{q['recall_elevated']:.3f}/{q['recall_critical_tier']:.3f}")


tp = pd.read_parquet(os.path.join(OUT, "predictions_test.parquet"))
summary = {
    "headline": [line(sp, mod) for sp in ("event80_20", "mission") for mod in ("persistence", REC, "lgbm_a13__train_organizer",
                                                                                "lgbm_all__train_kelvins")],
    "subpopulations": [line(sp, mod, g) for sp in ("mission",) for g in ("final_lt1d", "final_1to2d", "final_ge2d")
                       for mod in ("persistence", REC, "lgbm_all__train_kelvins")],
    "narrative": [
        "Population: organizer usable_for_training = 11,942 events (8,946 Safe / 2,162 Elevated / 834 Critical); our final-tier "
        "labels match the organizer file on all 13,154 events (0 mismatches on the usable set). Inputs: CDMs with ttca >= 2 d only, "
        "harness A13 (84) / ALL (233) features; on the 8,293 Kelvins events they are bit-identical to research/bakeoff features.",
        "Sub-populations by final-message ttca: final_lt1d 8,293 (551 crit; = Kelvins-eligible set), final_1to2d 599 (39 crit), "
        "final_ge2d 3,050 (244 crit). On final_ge2d the final message IS the latest input, so persistence (latest >= 2 d tier) "
        "is exact by construction (recall 1, 0 FA, macro-F1 1). Any skill claimed on the full organizer population is diluted/"
        "inflated by this 26 % slice; the forecasting problem lives in final_lt1d and final_1to2d.",
        "No 'is this the final message' feature (unknowable at prediction time). The models do see last_time_to_tca, which is "
        "legitimate. Test events all have their latest message at 2.0-6.9 d (median 2.19 d), i.e. they look like truncated events.",
        "Alert cut-off: per outer fold, inner 5-fold OOF on the training part only (KFold by event for 80/20, GroupKFold by mission "
        "for the mission split); threshold = argmin 10*missed + FA subject to inner recall >= 90 %. Persistence alert = latest tier "
        "is Critical; persistence_costcut = same rule applied to the ordinal latest tier (it picks >= Elevated: recall 0.98 but "
        "~4x the false alarms, cost/1000 ~200).",
        "Recommended: LightGBM ALL trained on the organizer population. vs persistence it cuts R=10 cost by ~30 % on both splits "
        f"(80/20: {p('event80_20', REC)['cost_per_1000']:.1f} vs {p('event80_20', 'persistence')['cost_per_1000']:.1f}; mission: "
        f"{p('mission', REC)['cost_per_1000']:.1f} vs {p('mission', 'persistence')['cost_per_1000']:.1f} per 1000 events) and "
        "raises critical recall from ~0.85-0.87 to ~0.94-0.95, at ~3.5x the false alarms. Macro-F1 of the argmax tier is a tie "
        "with persistence (persistence gets a free 1.0 on final_ge2d); on final_lt1d LightGBM is level on macro-F1 and much "
        "better on cost/recall/AUC.",
        "Training population: organizer (11,942) beats Kelvins-only (8,293) mainly through final_ge2d (macro-F1 ~0.97 vs ~0.86, "
        "fewer FA) and better Elevated recall overall (0.85-0.87 vs 0.77-0.78); on final_lt1d the two are statistically tied (by mission: Kelvins-trained macro-F1 0.879 vs 0.870, "
        "organizer-trained cost/1000 86 vs 94; both within CI). "
        "So train on the organizer population. ALL vs A13 (both organizer-trained) is within CI on cost; ALL has the higher "
        "macro-F1 and AUC and was the bake-off winner, so ALL is kept.",
        f"Final model fitted on all 11,942; alert threshold p_critical >= {f['threshold']:.4f} from a by-event 5-fold OOF on the "
        f"training data (OOF recall {f['threshold_info']['inner_recall']:.3f}). Test (2,167 events): predicted tiers "
        f"{f['test_pred_counts']}, {f['test_alerts']} CRITICAL alerts; persistence tiers {f['test_persistence_counts']}.",
        "CIs: event bootstrap (B=500); for 80/20 each of the 5 seed test sets is resampled independently and the metric averaged "
        "over seeds (counts like FA are per ~2,389-event test set); for mission the pooled OOF over all 11,942 is resampled. "
        "CIs are marginal per model, not paired differences.",
        "The cost-based sweep (research/sweep/SELECTION.md, registry/forecaster_cost/) had not produced a selection when this ran; "
        "its config is not included.",
    ],
}
res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "setup": {"R": 10, "recall_floor": 0.9, "recall_target": 0.95, "lgbm_params": "research/bigml/results.json best_lgbm_params",
                 "lgbm_seed": 0, "splits": {"event80_20": "GroupShuffleSplit(test_size=0.2, groups=event_id), random_state 0-4",
                                            "mission": "GroupKFold(5, groups=mission_id), 19 missions"},
                 "train_populations": {"organizer": 11942, "kelvins": 8293}, "bootstrap_B": 500},
       "build": b, "metrics": m, "final": f, "summary": summary}
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
card = {"name": "forecaster_organizer", "version": "1.0.0", "created": res["generated"],
        "task": "Predict each event's final_tier (organizer definition) from CDMs with time_to_tca >= 2 d; CRITICAL alert at R=10",
        "model": "LightGBM multiclass, ALL feature set (harness recipe)", "training_population": "organizer usable_for_training (11,942)",
        "params": f["params"], "n_features": f["n_features"], "alert_rule": f"p_critical >= {f['threshold']:.6f}",
        "alert_threshold": f["threshold"], "threshold_source": "by-event 5-fold OOF on training data, argmin 10*FN+FP s.t. recall>=0.9",
        "evaluation": {sp: {g: p(sp, REC, g) for g in ("all", "final_lt1d", "final_1to2d", "final_ge2d")} for sp in ("event80_20", "mission")},
        "baseline_persistence": {sp: p(sp, "persistence") for sp in ("event80_20", "mission")},
        "top_features_gain": f["top_features_gain"], "files": {"predictions_test.parquet": "2,167 test events",
        "oof_organizer.parquet": "by-mission outer OOF (+fold, per-fold threshold, alert) and by-event 5-fold OOF used for the threshold",
        "model.txt": "LightGBM booster"},
        "caveats": summary["narrative"][1:3], "source": "research/organizer/organizer.py"}
json.dump(card, open(os.path.join(REG, "model_card.json"), "w"), indent=1, default=float)
for fn in ("predictions_test.parquet", "oof_organizer.parquet", "model.txt"):
    shutil.copy(os.path.join(OUT, fn), os.path.join(REG, fn))
print("\n".join(summary["headline"] + summary["subpopulations"]))
