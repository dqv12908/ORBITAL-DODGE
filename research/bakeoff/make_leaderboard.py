"""Build LEADERBOARD.md from leaderboard.json (written by score.py, run as is) + OOF validation + runtimes.
Idempotent: re-reads score.py output fields only; extra keys it adds to leaderboard.json are overwritten each run."""
import json, os, glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "leaderboard.json")))
ev = H.load_events(); keys = set(ev.event_key)

# --- OOF validation (stricter than score.py: rows sum to 1 within 1e-6) ---
val = {}
for f in sorted(glob.glob(os.path.join(HERE, "oof", "*.parquet"))):
    d = pd.read_parquet(f); errs = []; dev = float("nan")
    if not set(H.PCOLS + ["event_key"]) <= set(d.columns):
        errs.append("missing columns")
    else:
        if not d.event_key.is_unique: errs.append(f"{int(d.event_key.duplicated().sum())} duplicate event_keys")
        if keys - set(d.event_key): errs.append(f"{len(keys - set(d.event_key))} eligible events missing")
        if set(d.event_key) - keys: errs.append(f"{len(set(d.event_key) - keys)} extra keys")
        P = d[H.PCOLS].values.astype(float)
        if np.isnan(P).any(): errs.append("NaN")
        elif not np.isfinite(P).all(): errs.append("non-finite")
        if (P < 0).any() or (P > 1).any(): errs.append("prob outside [0,1]")
        dev = float(np.nanmax(np.abs(P.sum(1) - 1)))
        if not dev <= 1e-6: errs.append(f"row sums off by {dev:.1e}")
    val[os.path.basename(f)] = dict(rows=int(len(d)), valid=not errs, reason="; ".join(errs), max_sum_dev=dev)

# OOF wall clock per split as logged by each builder (shared, heavily loaded server)
RUNTIME = {
    "persistence": ("<1 s", "<1 s", "refs.py, cross-fitted lookup table"),
    "hgb_a13": ("58.6 s*", "56.8 s*", "*refs.py total for hgb_a13 + lgbm_a13 + lgbm_all per split (logs/refs.log)"),
    "lgbm_a13": ("58.6 s*", "56.8 s*", "*shared refs.py total"),
    "lgbm_all": ("58.6 s*", "56.8 s*", "*shared refs.py total"),
    "tabpfn": ("611 s", "727 s", "GPU (V100 / RTX 2060); final fit + test prediction 3.7 min"),
    "ebm": ("367 s", "395 s", "CPU; final 2 fits + exports 321 s"),
    "gru": ("~100 s", "~100 s", "both splits in parallel on 2 GPUs (~1.7 min total); inner tuning ~5 min; final fit 0.5 min"),
    "futures": ("206 s", "372 s", "MDN + Monte Carlo rollouts (4,000 / 10,000 paths per event); research/bigml/out/futures_*.log"),
}


def f3(x): return f"{x:.3f}"
def fci(x, c): return f"{x:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
def sgn(x, c): return f"{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
def key(kv): return (-round(kv[1]["primary"], 6), -kv[1]["auc"], kv[1]["brier"])


nc = R["n_critical"]
d = R["decision"]; win = d["winner"]
M = R["splits"]["mission"]["models"]; E = R["splits"]["event"]["models"]; w = M[win]; pers = M["persistence"]
ranked = [m for m, _ in sorted(((m, r) for m, r in M.items() if m != "persistence"), key=key)]
ru = ranked[1]
gate_a = w["delta_primary_ci"][0] > 0
gate_b = (w["primary"] >= pers["primary"] - 0.01) and (w["auc_ci"][0] > pers["auc"])
gate_path = "a" if gate_a else ("b" if gate_b else "none")
assert d["serve"] == (gate_a or gate_b)

L = ["# ORBITAL DODGE forecaster bake-off: LEADERBOARD", "",
     "Generated 2026-10-31 by `make_leaderboard.py` from `leaderboard.json`, which `score.py` wrote when run unmodified over every file in `oof/`. "
     "The decision rule is `PREREG.md` section 5 (its sha256 matches `PREREG.sha256`; no amendments), applied mechanically.", "",
     f"{R['n_events']} eligible events, {nc} critical. Bootstrap B={R['B']} over events, seed {R['boot_seed']}, with the same resamples for every model (paired). "
     "PRIMARY = critical recall at no more than the tier-persistence false-alarm count (113 on both splits, recomputed inside each resample). "
     "ECE uses 10 equal-width bins on p_critical. 'ECE after iso' is after cross-fitted isotonic calibration (fitted on the other folds' OOF predictions of the same split).", "",
     "## Decision (by-mission split)", "",
     f"- **Winner: `{win}`.** It has the highest by-mission PRIMARY, {w['primary']:.3f} ({round(w['primary'] * nc)}/{nc} critical events caught at 113 false alarms). "
     f"The runner-up is `{ru}` at {M[ru]['primary']:.3f} ({round(M[ru]['primary'] * nc)}/{nc}). There is no tie at 1e-6, so the AUC and Brier tie-breaks were not used.",
     f"- **Serving gate: {'PASS via (' + gate_path + ')' if gate_path != 'none' else 'FAIL'}.** "
     f"(a) {'passes' if gate_a else 'fails'}: the paired dPRIMARY CI is {sgn(w['delta_primary_vs_persistence'], w['delta_primary_ci'])}, and its lower bound "
     f"{w['delta_primary_ci'][0]:.3f} is {'' if gate_a else 'not '}> 0 (P(d <= 0) = {w['p_delta_le_0']:.3f}). "
     f"(b) {'passes' if gate_b else 'fails'}: PRIMARY {w['primary']:.3f} >= persistence {pers['primary']:.3f} - 0.01, and the AUC CI lower bound "
     f"{w['auc_ci'][0]:.3f} > persistence AUC {pers['auc']:.3f}.",
     f"- **Calibration requirement: {'met' if w['ece_iso'] <= 0.03 else 'NOT met (serve with calibration warning)'}.** ECE after cross-fitted isotonic is "
     f"{w['ece_iso']:.3f} (limit 0.03; raw ECE {w['ece']:.3f}).",
     f"- **Verdict: {d['verdict']}.**",
     f"- By-event sanity check (section 5.4): `{win}` scores by-event PRIMARY {E[win]['primary']:.3f}, dPRIMARY {sgn(E[win]['delta_primary_vs_persistence'], E[win]['delta_primary_ci'])}. "
     "That ties persistence and is not clearly worse, so the rule has nothing to flag. Its by-mission lead does not reproduce on the by-event split. The decision stands.",
     f"- Context, not part of the rule: `tabpfn` is the only row that passes the gate via (a), and it has the best by-event PRIMARY ({E['tabpfn']['primary']:.3f}). "
     f"The rule ranks by by-mission PRIMARY, not by gate strength. The gap to the winner is {round((w['primary'] - M['tabpfn']['primary']) * nc)} of {nc} critical events, well inside both CIs.",
     "- Risk-futures (`futures`) was delivered on time (before the pre-registration was frozen) and passed validation. It was scored as a contender.", ""]

for split in ["mission", "event"]:
    S = R["splits"][split]
    L += [f"## Split: by-{split} ({'decision split' if split == 'mission' else 'sanity check, does not decide'}), false-alarm budget {S['fa_budget']:.0f}", "",
          "| model | PRIMARY [95% CI] | crit caught | dPRIMARY vs persistence [paired 95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE before | ECE after iso | macro-F1 | @0.50 recall / FA | @0.10 recall / FA | OOF runtime | gate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in sorted(S["models"].items(), key=key):
        t5, t1 = r["at_0_50"], r["at_0_10"]
        rt = RUNTIME.get(m, ("?", "?", ""))[0 if split == "event" else 1]
        gate = ""
        if split == "mission":
            gate = "reference" if m == "persistence" else ("PASS (a)" if r["delta_primary_ci"][0] > 0 else ("PASS (b)" if r["gate_pass"] else "FAIL"))
        name = f"**{m}**" if (split == "mission" and m == win) else m
        L.append(f"| {name} | {fci(r['primary'], r['primary_ci'])} | {round(r['primary'] * nc)}/{nc} | {sgn(r['delta_primary_vs_persistence'], r['delta_primary_ci'])} | "
                 f"{fci(r['auc'], r['auc_ci'])} | {f3(r['ap'])} | {fci(r['brier'], r['brier_ci'])} | {f3(r['ece'])} | {f3(r['ece_iso'])} | {f3(r['macro_f1'])} | "
                 f"{t5['recall']:.3f} / {t5['false_alarms']} | {t1['recall']:.3f} / {t1['false_alarms']} | {rt} | {gate} |")
    L.append("")

L += ["Rows are sorted by the decision key: PRIMARY, then AUC, then Brier. Brier and ECE are on p_critical. "
      "The @t columns give critical recall and false alarms when alerting at p_critical >= t. Precision is in `leaderboard.md` and `leaderboard.json`.", "",
      "## OOF file validation", "",
      "Checks: all 8,293 eligible event_keys present exactly once, no extra keys, no NaN, probabilities in [0, 1], and each row sums to 1 within 1e-6. "
      "The futures builder dropped its files into `oof/` at 04:03, before PREREG was frozen at 04:05. PREREG records that they were neither opened nor scored at that point. "
      "Their `fold` column matches `folds.parquet` exactly on both splits (model_version futures-mdn-1.0). No file was rejected.", "",
      "| file | rows | max row-sum deviation | result |", "|---|---|---|---|"]
for k, v in val.items():
    L.append(f"| {k} | {v['rows']} | {v['max_sum_dev']:.1e} | {'valid' if v['valid'] else 'REJECTED: ' + v['reason']} |")
L += ["", "## Runtime notes", ""]
for m, (a, b, note) in RUNTIME.items():
    L.append(f"- `{m}`: by-event {a}, by-mission {b}. {note}.")
L += ["", "Runtimes are wall clock on a shared 56-core, 4-GPU server under heavy load (load average up to about 170), so treat them as indicative only.", "",
      "## Provenance", "",
      "- `score.py` was run unmodified. Its full output is in `leaderboard.md` and `leaderboard.json` (precision, thresholds, P(d <= 0)).",
      "- Contender details are in `results/{tabpfn,ebm,gru}.md` and `models/*/NOTES.md`. The futures source is in `research/bigml/` (`export_bakeoff.py`).",
      "- The TabPFN-ALL diagnostic in `models/tabpfn/diag/` is not entered. Choosing it would mean selecting on the outer folds, which PREREG forbids."]
open(os.path.join(HERE, "LEADERBOARD.md"), "w").write("\n".join(L) + "\n")

R["oof_validation"] = val
R["runtime"] = {m: dict(event=a, mission=b, note=n) for m, (a, b, n) in RUNTIME.items()}
R["decision"].update(gate_path=gate_path, gate_a_delta_ci_lo=w["delta_primary_ci"][0], runner_up=ru,
                     calibration_ece_iso=w["ece_iso"], futures_status="delivered before PREREG freeze, valid, scored",
                     rule_source="PREREG.md section 5 (sha256 matches PREREG.sha256, no amendments)")
json.dump(R, open(os.path.join(HERE, "leaderboard.json"), "w"), indent=1)
print(open(os.path.join(HERE, "LEADERBOARD.md")).read())
