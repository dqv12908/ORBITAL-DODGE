# tier_forecaster 1.0.0

**Question:** at the latest CDM with time-to-TCA in [2, 7] d, what is P(final-CDM brief tier = CRITICAL)?
**Model:** sklearn `HistGradientBoostingClassifier` (3-class; 300 it, lr 0.05, 31 leaves, min leaf 20, seed 0) on the
84 A13 features (13 allowed columns only; no `risk`, no `mission_id`), isotonic calibration of P(critical).
**Data:** 8,293 Kelvins-eligible train events (≥ 1 CDM at ≥ 2 d, final CDM < 1 d), 551 final-critical.
**Use:** rank the queue and add a WATCH flag. It never lowers a rule tier or removes an escalation.
Built by `server/pipelines/train.py`. Full numbers: `metrics.json`, `reliability.json`, OOF predictions: `oof.parquet`.

## Headline (out-of-fold, 8,293 events)

| | by event (KFold 5) | by mission (GroupKFold 5) |
|---|---|---|
| **Tier persistence** (latest ≥ 2 d tier): macro-F1 / crit recall / false alarms / missed crit | 0.868 / **0.820** / 113 / **99** | 0.868 / **0.820** / 113 / **99** |
| **HGB argmax** (raw probabilities) | **0.871** / 0.779 / **80** / 122 | **0.872** / 0.775 / **78** / 124 |
| HGB argmax after isotonic on P(crit) | 0.868 / 0.739 / 59 / 144 | 0.868 / 0.764 / 85 / 130 |
| P(crit) AUC: HGB calibrated (raw) vs persistence-empirical | **0.978** (0.981) vs 0.958 | **0.980** (0.983) vs 0.958 |
| P(crit) Brier: HGB calibrated vs persistence-empirical vs climatology | **0.0193** vs 0.0223 vs 0.0620 | **0.0192** vs 0.0223 vs 0.0620 |
| P(crit) ECE (10 bins): HGB raw → calibrated; persistence-empirical | 0.021 → 0.004; 0.003 | 0.020 → 0.004; 0.001 |
| Persistence CRITICAL **or** WATCH (calibrated p ≥ 0.10): crit recall / alerts / false alarms | 0.922 / 922 / 414 | 0.906 / 840 / 341 |
| Kelvins metric, LRP baseline (shipped as the "final risk" value) | L = 0.804 (F2 0.411, MSE_HR 0.330) | same (no fitting) |

The by-mission check passes the pre-registered rule (macro-F1 drop ≤ 0.05, AUC ≥ persistence): it is the same as
by event (drop −0.002). Whole-mission hold-out does not hurt this model.

## Where it loses (read this)

- **As an alarm it does not beat persistence.** At argmax it misses **23–25 more criticals** than tier persistence
  (122 vs 99 by event). It cuts false alarms only by moving to a lower-recall point. On raw scores at τ = 0.10 it is a tie
  (recall 0.819 vs 0.820, 119 vs 113 false alarms). That is why it ranks and promotes and never gates an alert.
- **Calibration helps probabilities but hurts the argmax.** Isotonic fixes the overconfident top bin and the
  under-confident low bins (ECE 0.021 → 0.004), but the calibrated argmax misses more criticals (144). The product uses
  calibrated p for display and WATCH and **raw argmax is not used to decide anything**.
- **WATCH at τ = 0.10 is generous.** Calibrated p ≥ 0.10 promotes 357 extra events (4.3 %) to catch 56 more
  criticals (recall 0.820 → 0.922). That is a lot of extra attention. τ_watch is a desk knob; τ = 0.20 alone gives
  recall 0.853 with 656 alerts.
- **Late escalations are not foreseeable.** Example: train:2563 is SAFE at T−2.43 d (p = 0.005), CRITICAL at T−0.44 d.
  The worst miss with Pc-critical at the end is train:11180 (persistence ELEVATED, p = 0.001, final Pc 1.3e-4 at 642 m):
  it is the `failure` scenario.
- **Labels:** final-CDM brief tier from the dataset `risk` and miss. Kelvins pre-filtering removed manoeuvre epochs, so
  "resolved" events may be manoeuvred ones (Uriot §3). The final-risk regressor is not shipped: LRP wins (11 §B).
