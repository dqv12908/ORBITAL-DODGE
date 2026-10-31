"""C: sequential decision policy (MONITOR / REQUEST_TRACKING / PREPARE_MANOEUVRE) at each CDM, solved by backward
induction (value iteration) on a discretised CDM state, with transitions estimated from real CDM sequences and from the
MDN's simulated futures. Evaluated on held-out missions (GroupKFold by mission) against rule policies.

Loss assumptions (analyst-hour equivalents, extend design §4.4 to a sequence; editable, stated not measured):
  PREPARE at ttca >= 1 d (before the commitment window): 5 if final CRITICAL else 8; terminal.
  PREPARE at ttca <  1 d (inside the window, rushed):    +LATE surcharge (10) on top.
  REQUEST_TRACKING: 1 per request; effect = information arrives one CDM early (the next decision sees the state of the
     CDM after next, at the next CDM's time) - a proxy for 'tracking reduces sigma / the estimate converges sooner'.
  MONITOR: 0; if the event ends (last CDM passed) un-prepared and the final CDM is CRITICAL: 100 (a missed critical).
"""
import os, json, joblib, numpy as np, pandas as pd
from bm import HERE, ROOT, load_rows, events_table, folds, tier

OUT = os.path.join(HERE, "out")
REG = os.path.join(ROOT, "registry", "policy"); os.makedirs(REG, exist_ok=True)
COST = dict(prep_crit=5.0, prep_safe=8.0, late=10.0, track=1.0, miss=100.0, mcp_d=1.0)
TB = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5]            # ttca bin edges (descending) -> 7 bins
MB = [300, 1000, 2000, 5000, 10000, 30000]     # miss edges (m) -> 7 bins
RB = [-4.0, -6.0, -10.0]                       # risk edges -> 4 bins
TR = [-0.1, 0.1]                               # d log10 miss vs previous CDM -> 3 bins
NT, NM, NR, NTR = 7, 7, 4, 3
NS = NT * NM * NR * NTR
ACT = ["MONITOR", "REQUEST_TRACKING", "PREPARE_MANOEUVRE"]


def tbin(t): return np.searchsorted(-np.array(TB), -np.asarray(t), side="right")
def mbin(m): return np.searchsorted(np.array(MB), np.asarray(m), side="right")
def rbin(r): return np.searchsorted(-np.array(RB), -np.asarray(r), side="left")
def trbin(dl): return np.searchsorted(np.array(TR), np.asarray(dl), side="right")
def sid(tb, mb, rb, trb): return ((tb * NM + mb) * NR + rb) * NTR + trb


def info_state(t, m, r, dl):
    return tbin(t), mbin(m), rbin(r), trbin(dl)


def seqs_real(d, keys):
    """per event: arrays ttca, miss, risk, dlm and the final-critical label"""
    out = []
    for k, e in d[d.event_key.isin(keys)].groupby("event_key", sort=False):
        t = e.time_to_tca.values; m = e.miss_distance.values.clip(1); r = e.risk.values.clip(-30, 0)
        dl = np.r_[0.0, np.diff(np.log10(m))]
        crit = int(tier(r[-1], m[-1]) == 2)
        out.append((k, t, m, r, dl, crit))
    return out


def seqs_sim(paths, keys_mask):
    idx = np.where(keys_mask)[0]
    T, L, R = paths["T"][idx], paths["L"][idx].astype(np.float32), paths["R"][idx].astype(np.float32)
    n = (T > -1e8).sum(1)
    out = []
    for i in range(len(idx)):
        k = n[i]
        if k < 2: continue
        l = L[i, :k]; m = 10 ** l; r = R[i, :k]
        out.append(("sim", T[i, :k], m, r, np.r_[0.0, np.diff(l)], int((r[-1] >= -4) | (m[-1] < 1000))))
    return out


def build(seqs, w=1.0):
    """transition counts: monitor s->s' (or END), track s->s'' (time of k+1, info of k+2), plus final-crit stats."""
    Cm = np.zeros((NS, NS + 1)); Ct = np.zeros((NS, NS + 1))
    if not seqs:
        z = np.zeros(NS); return Cm, Ct, z, z.copy(), z.copy()
    t = np.concatenate([s[1] for s in seqs]).astype(float); m = np.concatenate([s[2] for s in seqs]).astype(float)
    r = np.concatenate([s[3] for s in seqs]).astype(float); dl = np.concatenate([s[4] for s in seqs]).astype(float)
    lens = np.array([len(s[1]) for s in seqs]); c = np.repeat(np.array([s[5] for s in seqs], float), lens)
    pos = np.concatenate([np.arange(L) for L in lens]); K = np.repeat(lens, lens)
    S = sid(*info_state(t, m, r, dl))
    visits = np.bincount(S, minlength=NS) * w; crit_n = np.bincount(S, weights=c, minlength=NS) * w
    last = pos == K - 1
    endcrit = np.bincount(S[last], weights=c[last], minlength=NS) * w
    np.add.at(Cm[:, NS], S[last], w); np.add.at(Ct[:, NS], S[last], w)
    nl = np.where(~last)[0]
    np.add.at(Cm, (S[nl], S[nl + 1]), w)
    has2 = pos[nl] + 2 < K[nl]
    i2 = nl[has2]
    s2 = sid(tbin(t[i2 + 1]), mbin(m[i2 + 2]), rbin(r[i2 + 2]), trbin(np.log10(m[i2 + 2]) - np.log10(m[i2])))
    np.add.at(Ct, (S[i2], s2), w)
    i1 = nl[~has2]
    np.add.at(Ct, (S[i1], S[i1 + 1]), w)
    return Cm, Ct, endcrit, crit_n, visits


def pcrit_state(crit_n, visits, prior=5.0):
    """P(final CRITICAL | state) with backoff to the (ttca bin, tier) average."""
    gp = np.zeros(NS)
    idx = np.arange(NS); trb = idx % NTR; rb = (idx // NTR) % NR; mb = (idx // (NTR * NR)) % NM; tb = idx // (NTR * NR * NM)
    tr_ = np.where((mb <= 1) | (rb == 0), 2, np.where((mb <= 3) | (rb == 1), 1, 0))
    for a in range(NT):
        for b in range(3):
            g = (tb == a) & (tr_ == b)
            v = visits[g].sum(); gp[g] = (crit_n[g].sum() + 0.5) / (v + 1.0) if v > 0 else 0.5
    return (crit_n + prior * gp) / (visits + prior), tb


def solve(Cm, Ct, endcrit, crit_n, visits, cost=COST, iters=200):
    pc, tb = pcrit_state(crit_n, visits)
    late = (tb >= 5)  # bins 1-0.5 and <0.5 d are inside the commitment window
    prep = pc * cost["prep_crit"] + (1 - pc) * cost["prep_safe"] + late * cost["late"]
    def norm(C):
        tot = C.sum(1, keepdims=True); P = np.where(tot > 0, C / np.maximum(tot, 1e-9), 0)
        P[tot[:, 0] == 0, NS] = 1.0   # unseen state: assume END
        return P
    Pm, Pt = norm(Cm), norm(Ct)
    # end cost: probability of final crit given END after state s
    tot_end = Cm[:, NS]
    pend = np.where(tot_end > 0, (endcrit + 5 * pc) / (tot_end + 5), pc)
    V = prep.copy()
    for _ in range(iters):
        Vx = np.r_[V, 0.0]
        qm = Pm[:, :NS] @ V + Pm[:, NS] * pend * cost["miss"]
        qt = cost["track"] + Pt[:, :NS] @ V + Pt[:, NS] * pend * cost["miss"]
        Vn = np.minimum(np.minimum(qm, qt), prep)
        if np.abs(Vn - V).max() < 1e-6: V = Vn; break
        V = Vn
    Q = np.stack([qm, qt, prep], 1)
    return Q.argmin(1), Q, pc


def simulate(policy_fn, seqs, cost=COST):
    """run a policy over real held-out sequences (with the tracking information-lead mechanic)."""
    tot = []; missed = 0; preps = 0; tracks = 0; prep_noncrit = 0; prep_late = 0
    for k_, t, m, r, dl, c in seqs:
        K = len(t); i = 0; j = 0; L = 0.0; done = False
        while i < K:
            dlj = np.log10(m[j]) - np.log10(m[j - 1]) if j > 0 else 0.0
            a = policy_fn(t[i], m[j], r[j], dlj)
            if a == 2:
                late = t[i] < cost["mcp_d"]
                L += (cost["prep_crit"] if c else cost["prep_safe"]) + late * cost["late"]
                preps += 1; prep_noncrit += (1 - c); prep_late += late; done = True; break
            if a == 1:
                L += cost["track"]; tracks += 1
                i += 1; j = min(max(j, i) + 1, K - 1)
            else:
                i += 1; j = max(j, i)
        if not done and c:
            L += cost["miss"]; missed += 1
        tot.append(L)
    tot = np.array(tot)
    return dict(mean_loss=float(tot.mean()), total_loss=float(tot.sum()), missed_crit=int(missed), prepares=int(preps),
                prepares_noncrit=int(prep_noncrit), prepares_late=int(prep_late), tracking_requests=int(tracks), n=len(seqs)), tot


def stier(r, m):
    return 2 if (r >= -4 or m < 1000) else (1 if ((-6 <= r < -4) or (1000 <= m <= 5000)) else 0)
def rule_brief(t, m, r, dl):
    return stier(r, m)
def rule_prep_if_crit(t, m, r, dl):
    return 2 if stier(r, m) == 2 else 0
def rule_oneshot(t, m, r, dl):
    """design §4.4 one-shot table with persistence-empirical p (safe .002, elevated .053, critical .800)"""
    p = {0: 0.002, 1: 0.053, 2: 0.800}[stier(r, m)]
    q = [100 * p, 30 * p + 1 * (1 - p), 5 * p + 8 * (1 - p)]
    return int(np.argmin(q))


def dp_fn(pol):
    def f(t, m, r, dl):
        return int(pol[sid(tbin(t), mbin(m), rbin(r), trbin(dl))])
    return f


if __name__ == "__main__":
    d = load_rows(); ev = events_table(d)
    el = ev[(ev.split == "train") & ev.eligible].reset_index(drop=True)
    evtr = ev[ev.split == "train"]
    F = joblib.load(os.path.join(OUT, "futures_mission.joblib"))
    fk = np.array(F["event_key"]); fold_of = dict(zip(fk, F["fold"]))
    P = F["paths"]; P_ev = np.repeat(fk, 32)
    import time; t0 = time.time()
    real_all = {s[0]: s for s in seqs_real(d, set(evtr.event_key))}
    print("real seqs", len(real_all), f"{time.time()-t0:.0f}s", flush=True)
    res = {"cost_assumptions": COST, "folds": [], "policies": {}}
    agg = {k: [] for k in ["dp_real", "dp_real_sim", "brief_rule", "prepare_if_critical", "oneshot_4_4", "oracle", "never"]}
    for fi, (a, b) in enumerate(folds(el, "mission")):
        test_m = set(el.mission_id.iloc[b])
        tr_keys = set(evtr.event_key[~evtr.mission_id.isin(test_m)])
        te_seqs = [real_all[k] for k in el.event_key.iloc[b]]
        tr_seqs = [real_all[k] for k in tr_keys]
        Cm, Ct, ec, cn, vi = build(tr_seqs)
        pol_r, Q_r, pc_r = solve(Cm, Ct, ec, cn, vi)
        # + simulated futures of train-fold events (MDN paths, weight 1/8 per path so 32 paths ~ 4 real sequences)
        mask = pd.Series(P_ev).isin(tr_keys).values
        sim = seqs_sim(P, mask)
        Cm2, Ct2, ec2, cn2, vi2 = build(sim, w=1 / 8)
        pol_s, Q_s, _ = solve(Cm + Cm2, Ct + Ct2, ec + ec2, cn + cn2, vi + vi2)
        for name, fn in [("dp_real", dp_fn(pol_r)), ("dp_real_sim", dp_fn(pol_s)), ("brief_rule", rule_brief),
                         ("prepare_if_critical", rule_prep_if_crit), ("oneshot_4_4", rule_oneshot), ("never", lambda *x: 0)]:
            agg[name].append(simulate(fn, te_seqs)[1])
        agg["oracle"].append(np.array([COST["prep_crit"] if s[5] else 0.0 for s in te_seqs]))
        print(f"[{time.time()-t0:.0f}s] fold {fi} missions {sorted(test_m)}: " + " ".join(f"{k}={np.mean(v[-1]):.3f}" for k, v in agg.items()))
    # pooled held-out results with bootstrap CI
    rng = np.random.default_rng(0)
    for name in agg:
        tot = np.concatenate(agg[name])
        bs = [tot[rng.integers(0, len(tot), len(tot))].mean() for _ in range(1000)]
        res["policies"][name] = dict(mean_loss=float(tot.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    # detailed counters (pooled) for the main policies
    all_te = [real_all[k] for k in el.event_key]
    # final policy fitted on all train (export) + in-fold detail counters by re-running per fold
    det = {k: dict(missed_crit=0, prepares=0, prepares_noncrit=0, prepares_late=0, tracking_requests=0) for k in ["dp_real", "dp_real_sim", "brief_rule", "prepare_if_critical", "oneshot_4_4"]}
    for fi, (a, b) in enumerate(folds(el, "mission")):
        test_m = set(el.mission_id.iloc[b]); tr_keys = set(evtr.event_key[~evtr.mission_id.isin(test_m)])
        te_seqs = [real_all[k] for k in el.event_key.iloc[b]]
        Cm, Ct, ec, cn, vi = build([real_all[k] for k in tr_keys]); pol_r = solve(Cm, Ct, ec, cn, vi)[0]
        sim = seqs_sim(P, pd.Series(P_ev).isin(tr_keys).values); Cm2, Ct2, ec2, cn2, vi2 = build(sim, w=1 / 8)
        pol_s = solve(Cm + Cm2, Ct + Ct2, ec + ec2, cn + cn2, vi + vi2)[0]
        for name, fn in [("dp_real", dp_fn(pol_r)), ("dp_real_sim", dp_fn(pol_s)), ("brief_rule", rule_brief),
                         ("prepare_if_critical", rule_prep_if_crit), ("oneshot_4_4", rule_oneshot)]:
            r_, _ = simulate(fn, te_seqs)
            for kk in det[name]: det[name][kk] += r_[kk]
    for k in det: res["policies"][k].update(det[k])
    res["policies"]["oracle"]["note"] = "perfect foresight: prepare at first CDM iff final critical (lower bound)"
    res["n_heldout_events"] = len(el); res["n_crit"] = int(sum(s[5] for s in all_te))
    # export policy on all train (real + simulated)
    Cm, Ct, ec, cn, vi = build(list(real_all.values()))
    Cm2, Ct2, ec2, cn2, vi2 = build(seqs_sim(P, np.ones(len(P_ev), bool)), w=1 / 8)
    pol, Q, pc = solve(Cm + Cm2, Ct + Ct2, ec + ec2, cn + cn2, vi + vi2)
    _, _, _, _, vis = build(list(real_all.values()))
    table = []
    for s in range(NS):
        trb = s % NTR; rb = (s // NTR) % NR; mb = (s // (NTR * NR)) % NM; tb = s // (NTR * NR * NM)
        table.append(dict(state=s, ttca_bin=int(tb), miss_bin=int(mb), risk_bin=int(rb), trend_bin=int(trb), action=ACT[pol[s]],
                          q={ACT[i]: round(float(Q[s, i]), 3) for i in range(3)}, p_final_critical=round(float(pc[s]), 4),
                          n_real_visits=int(vis[s])))
    policy = dict(model_version="policy-dp-1.0", cost_assumptions=COST, actions=ACT,
                  bins=dict(ttca_d_desc_edges=TB, miss_m_edges=MB, risk_log10_desc_edges=RB, dlog10_miss_edges=TR),
                  state_index="((ttca_bin*7 + miss_bin)*4 + risk_bin)*3 + trend_bin",
                  how_to_bin=("ttca_bin = number of ttca edges (descending) that are > ttca [i.e. searchsorted(-edges,-ttca,'right')]; "
                              "miss_bin = searchsorted(miss_edges, miss_m, 'right'); risk_bin = searchsorted(-risk_edges, -risk, 'right'); "
                              "trend_bin = searchsorted(dlog10_miss_edges, log10(miss)-log10(prev miss), 'right') (0 change for the first CDM)"),
                  semantics="Decision aid benchmark only. Picks among actions to minimise stated expected loss; never overrides rule gating (design §4.2).",
                  table=table)
    json.dump(policy, open(os.path.join(REG, "policy.json"), "w"), indent=0)
    json.dump(res, open(os.path.join(REG, "metrics.json"), "w"), indent=1)
    print(json.dumps(res["policies"], indent=1))
