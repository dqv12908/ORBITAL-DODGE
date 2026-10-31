"""Train the tier forecaster (HGB A13, isotonic-calibrated P(critical)) and write registry/tier_forecaster/1.0.0/.

OOF by event (KFold 5, shuffle, seed 0) and by mission (GroupKFold 5); baselines: tier persistence,
persistence-empirical P(crit), climatology, LRP for the Kelvins metric.
"""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import datetime as dt
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold

from riskengine import evaluate as ev
from riskengine import paths
from riskengine.cdm import TIER_NAMES
from riskengine.features import FEATURES, event_features, kelvins_frame
from riskengine.forecast import TAU_WATCH, calibrate_tiers

VERSION = "1.0.0"
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=0)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hgb():
    return HistGradientBoostingClassifier(**HGB_PARAMS)


def iso():
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)


def run_split(X, yt, ycrit, pers, folds):
    raw = np.zeros((len(yt), 3))
    fold_id = np.zeros(len(yt), int)
    for k, (trn, tst) in enumerate(folds):
        m = hgb().fit(X[trn], yt[trn])
        raw[tst] = m.predict_proba(X[tst])
        fold_id[tst] = k
    # cross-fitted isotonic: calibrator for fold k fitted on the raw OOF of the other folds
    cal = np.zeros_like(raw)
    for k, (trn, tst) in enumerate(folds):
        c = iso().fit(raw[trn, 2], ycrit[trn])
        cal[tst] = calibrate_tiers(raw[tst], c)
    # persistence-empirical P(crit | latest tier), in-fold
    p_pers = np.zeros(len(yt))
    for trn, tst in folds:
        for k in range(3):
            mtr = pers[trn] == k
            p_pers[tst[pers[tst] == k]] = ycrit[trn][mtr].mean() if mtr.any() else 0.0
    return raw, cal, p_pers, fold_id


def split_metrics(yt, ycrit, pers, raw, cal, p_pers, latest_risk, y_risk):
    pc_raw, pc_cal = raw[:, 2], cal[:, 2]
    watch = (pc_cal >= TAU_WATCH) & (pers != 2)
    pers_or_watch = (pers == 2) | watch
    lrp = np.where(latest_risk >= -6, latest_risk, -6.001)
    return {
        "n_events": int(len(yt)), "n_final_critical": int(ycrit.sum()),
        "tier_persistence": ev.tier_report(yt, pers),
        "persistence_empirical_pcrit": ev.prob_report(ycrit, p_pers),
        "climatology_brier": float(np.mean((ycrit - ycrit.mean()) ** 2)),
        "hgb_argmax": ev.tier_report(yt, raw.argmax(1)),
        "hgb_argmax_calibrated": ev.tier_report(yt, cal.argmax(1)),
        "hgb_pcrit_raw": ev.prob_report(ycrit, pc_raw),
        "hgb_pcrit_calibrated": ev.prob_report(ycrit, pc_cal),
        "operating_curve_calibrated": ev.operating_curve(ycrit, pc_cal),
        "operating_curve_raw": ev.operating_curve(ycrit, pc_raw),
        "persistence_or_watch": {
            "tau_watch": TAU_WATCH, "alerts": int(pers_or_watch.sum()),
            "crit_recall": float((pers_or_watch & (ycrit == 1)).sum() / ycrit.sum()),
            "false_alarms": int((pers_or_watch & (ycrit == 0)).sum()),
            "missed_crit": int((~pers_or_watch & (ycrit == 1)).sum()),
            "extra_alerts_vs_persistence": int(watch.sum()),
            "extra_criticals_caught": int((watch & (ycrit == 1)).sum()),
        },
        "kelvins_lrp": ev.kelvins(y_risk, lrp),
    }


def main():
    t0 = time.time()
    df = pd.read_parquet(paths.CDM_CLEAN)
    tr = df[df.split == "train"]
    inp, tgt = kelvins_frame(tr)
    F = event_features(inp).loc[tgt.index]
    X = F.to_numpy(float)
    yt = tgt.y_tier.to_numpy()
    ycrit = (yt == 2).astype(int)
    pers = tgt.persistence_tier.to_numpy()
    print(f"[train] {len(tgt)} eligible events, {ycrit.sum()} final-critical, {X.shape[1]} features ({time.time()-t0:.1f}s)")

    splits = {
        "by_event": list(KFold(5, shuffle=True, random_state=0).split(X)),
        "by_mission": list(GroupKFold(5).split(X, groups=tgt.mission_id.to_numpy())),
    }
    metrics, oof = {}, pd.DataFrame(index=tgt.index)
    oof["mission_id"] = tgt.mission_id
    oof["event_id"] = tgt.event_id
    oof["y_tier"] = [TIER_NAMES[i] for i in yt]
    oof["y_log10_risk"] = tgt.y_risk
    oof["y_miss_m"] = tgt.y_miss
    oof["persistence_tier"] = [TIER_NAMES[i] for i in pers]
    oof["latest_ttca"] = tgt.latest_ttca
    oof["latest_risk"] = tgt.latest_risk
    rel = {}
    for name, folds in splits.items():
        raw, cal, p_pers, fold_id = run_split(X, yt, ycrit, pers, folds)
        metrics[name] = split_metrics(yt, ycrit, pers, raw, cal, p_pers, tgt.latest_risk.to_numpy(), tgt.y_risk.to_numpy())
        rel[name] = {"raw": ev.reliability(ycrit, raw[:, 2]), "calibrated": ev.reliability(ycrit, cal[:, 2]),
                     "persistence_empirical": ev.reliability(ycrit, p_pers)}
        sfx = "event" if name == "by_event" else "mission"
        oof[f"fold_{sfx}"] = fold_id
        oof[f"p_crit_raw_{sfx}"] = raw[:, 2]
        oof[f"p_safe_{sfx}"], oof[f"p_elevated_{sfx}"], oof[f"p_critical_{sfx}"] = cal[:, 0], cal[:, 1], cal[:, 2]
        oof[f"p_pers_emp_{sfx}"] = p_pers
        m = metrics[name]
        print(f"[train] {name}: persistence F1 {m['tier_persistence']['macro_f1']:.3f} rec {m['tier_persistence']['rec_crit']:.3f} "
              f"FA {m['tier_persistence']['false_alarms']} | HGB F1 {m['hgb_argmax']['macro_f1']:.3f} rec {m['hgb_argmax']['rec_crit']:.3f} "
              f"FA {m['hgb_argmax']['false_alarms']} | AUC hgb {m['hgb_pcrit_calibrated']['auc']:.3f} (raw {m['hgb_pcrit_raw']['auc']:.3f}) "
              f"vs pers {m['persistence_empirical_pcrit']['auc']:.3f} | Brier {m['hgb_pcrit_calibrated']['brier']:.4f} "
              f"ECE {m['hgb_pcrit_calibrated']['ece']:.4f} ({time.time()-t0:.1f}s)")

    # pre-registered acceptance for the by-mission split
    de = metrics["by_event"]["hgb_argmax"]["macro_f1"] - metrics["by_mission"]["hgb_argmax"]["macro_f1"]
    auc_m = metrics["by_mission"]["hgb_pcrit_calibrated"]["auc"]
    auc_p = metrics["by_mission"]["persistence_empirical_pcrit"]["auc"]
    generalises = bool(de <= 0.05 and auc_m >= auc_p)
    metrics["acceptance_by_mission"] = {"macro_f1_drop": de, "auc_model": auc_m, "auc_persistence": auc_p,
                                        "generalises": generalises,
                                        "rule": "macro-F1 drop <= 0.05 and AUC >= persistence-empirical"}

    # final model + calibrator (isotonic on all by-event OOF raw P(crit))
    model = hgb().fit(X, yt)
    calibrator = iso().fit(oof["p_crit_raw_event"].to_numpy(), ycrit)
    # persistence-empirical transition on all eligible events
    pemp = {TIER_NAMES[k]: float(ycrit[pers == k].mean()) for k in range(3)}

    out = paths.FORECASTER_DIR / VERSION
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(calibrator, out / "calibrator.joblib")
    oof.reset_index().rename(columns={"index": "event_key"}).to_parquet(out / "oof.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "reliability.json").write_text(json.dumps(rel, indent=1))
    card = {
        "name": "tier_forecaster", "version": VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "artifacts": {"model.joblib": sha256(out / "model.joblib"), "calibrator.joblib": sha256(out / "calibrator.joblib")},
        "training_data": {"file": "data/processed/cdm_clean.parquet", "sha256": sha256(paths.CDM_CLEAN),
                          "filter": "split=train; Kelvins-eligible events (>=1 CDM at ttca>=2 d, final CDM < 1 d); inputs = CDMs at ttca>=2 d",
                          "n_events": int(len(tgt)), "n_final_critical": int(ycrit.sum())},
        "label": "brief tier of the final CDM from dataset risk and miss distance (CRITICAL: risk>=-4 or miss<1 km; "
                 "ELEVATED: risk in [-6,-4) or miss in [1,5] km; else SAFE). Q-COV-DEFAULT floor not applied to labels.",
        "model": {"type": "sklearn HistGradientBoostingClassifier (3-class)", "params": HGB_PARAMS,
                  "calibration": "isotonic on P(critical), fitted on by-event OOF raw predictions; OOF metrics use cross-fitted calibration (fit on 4 folds, apply to the 5th)"},
        "features": FEATURES,
        "libraries": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__},
        "splits": {"by_event": "KFold(5, shuffle=True, random_state=0)", "by_mission": "GroupKFold(5) on mission_id"},
        "metrics_summary": {k: {"persistence": metrics[k]["tier_persistence"], "hgb_argmax": metrics[k]["hgb_argmax"],
                                "auc": {"hgb": metrics[k]["hgb_pcrit_calibrated"]["auc"],
                                        "persistence": metrics[k]["persistence_empirical_pcrit"]["auc"]}}
                            for k in splits},
        "acceptance_by_mission": metrics["acceptance_by_mission"],
        "intended_use": "rank the queue and promote WATCH (p >= tau_watch) on non-critical events; never suppress or demote a rule tier",
        "valid_window_ttca_d": [2, 7],
        "known_limits": [
            "F1 late escalation not foreseeable at T-2 d (safe->critical in 10 of 6,040 safe-at-T-2 d events)",
            "F5 at argmax the model misses more criticals than tier persistence; it can add attention, never remove it",
            "F6 labels biased by Kelvins pre-filtering around manoeuvre epochs",
            "trained on 2015-2019 ESA CDMs for 19 missions; mission 13 is train-only",
            "only the 13 allowed columns are used (no risk, no mission_id)",
        ],
        "runtime": {"features": FEATURES, "persistence_empirical": pemp, "tau_watch": TAU_WATCH,
                    "reliability_calibrated": rel["by_event"]["calibrated"], "valid_window_ttca_d": [2, 7]},
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    (paths.FORECASTER_DIR / "CURRENT").write_text(VERSION + "\n")
    print(f"[train] wrote {out} (generalises by mission: {generalises}) ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
