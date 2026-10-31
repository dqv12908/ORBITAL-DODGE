import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from pc2d import *
from scipy.stats import pearsonr, spearmanr
raw = load("train")
sent = (raw.t_sigma_r>6e7)|(raw.c_sigma_r>6e7)
print("SENTINEL rows (t or c sigma_r ~ 63,781,363):", int(sent.sum()), "events:", raw[sent].event_id.nunique(), "| t-side", int((raw.t_sigma_r>6e7).sum()), "c-side", int((raw.c_sigma_r>6e7).sum()))
print("  risk on sentinel rows:", raw[sent].risk.describe().round(2).to_dict())
print("  object types on sentinel rows:", raw[sent].c_object_type.value_counts().to_dict())
print("  sentinel in final CDM of events:", int(sent[raw.sort_values('time_to_tca').groupby('event_id').head(1).index].sum()))
df = raw[~sent & raw.c_sigma_r.notna()].copy()
for c in SIGMA_COLS: df = df[df[c] < 1e7]
print("clean rows:", len(df))
res = {}
for rot in [False, True]:
    for hbr in [5, 10, 20]:
        res[(rot,hbr)] = pc2d_log10(df, hbr, rotate=rot)[0]
y = df.risk.values; m_all = y > -30; m6 = y >= -10
print("\nrows with risk>-30:", m_all.sum(), " risk>=-10:", m6.sum(), " risk>=-6:", (y>=-6).sum())
for (rot,hbr), lp in res.items():
    lpf = np.maximum(lp, -30); off = np.median(y[m6]-lp[m6])
    print(f"rotate={rot!s:5} HBR={hbr:>2} m: pearson(risk>-30)={pearsonr(lpf[m_all],y[m_all])[0]:.4f} spearman={spearmanr(lpf[m_all],y[m_all])[0]:.4f} "
          f"MAE(risk>=-10)={np.mean(np.abs(lp[m6]-y[m6])):.3f} median(risk-pc)={off:+.3f} -> implied HBR={hbr*10**(off/2):.2f} m ; agree(-30 floor)={np.mean((lp<=-30)==(y<=-30)):.4f}")
lp5 = res[(True,5)]; off = np.median(y[m6]-lp5[m6]); hbr_star = 5*10**(off/2)
lpb = pc2d_log10(df, hbr_star, rotate=True)[0]; e = lpb - y
def summ(e, tag):
    print(f"{tag}: MAE log10 (risk>=-10)={np.mean(np.abs(e[m6])):.3f}, median |err|={np.median(np.abs(e[m6])):.3f}, p90 |err|={np.quantile(np.abs(e[m6]),.9):.3f}, within 0.1 dex: {np.mean(np.abs(e[m6])<0.1):.3f}, within 0.5 dex: {np.mean(np.abs(e[m6])<0.5):.3f}, spearman(risk>-30)={spearmanr(np.maximum(e+y,-30)[m_all], y[m_all])[0]:.4f}")
summ(e, f"\nGLOBAL HBR*={hbr_star:.2f} m")
# per-mission HBR: fit on even event_ids, evaluate on odd (split-half)
fit = (df.event_id.values % 2 == 0); hbr_m = {}
for mid, g in df[fit & m6].groupby("mission_id"):
    o = np.median(g.risk.values - lp5[(fit & m6) & (df.mission_id.values==mid)])
    hbr_m[mid] = 5*10**(o/2)
print("per-mission HBR (fit on even event_ids):", {k: round(v,2) for k,v in hbr_m.items()})
lpm = np.full(len(df), np.nan)
for mid, h in hbr_m.items():
    mm = df.mission_id.values==mid
    lpm[mm] = lp5[mm] + 2*np.log10(h/5)   # small-HBR scaling Pc ~ HBR^2
em = lpm - y; ev = ~fit
summ(np.where(ev, em, np.nan) if False else em, "PER-MISSION HBR (all rows)")
print(f"   held-out odd event_ids only: MAE(risk>=-10)={np.nanmean(np.abs(em[ev & m6])):.3f}, median |err|={np.nanmedian(np.abs(em[ev & m6])):.3f}, within 0.1 dex={np.nanmean(np.abs(em[ev & m6])<0.1):.3f}")
for lo,hi in [(-30,-20),(-20,-10),(-10,-6),(-6,-4),(-4,0)]:
    mm = ((y>=lo) if lo>-30 else (y>-30)) & (y<hi)
    print(f"   risk in [{lo},{hi}): n={mm.sum()} MAE={np.nanmean(np.abs(em[mm])):.3f} bias={np.nanmean(em[mm]):+.3f}")
for thr in [-6,-4]:
    yb=y>=thr; pb=lpm>=thr
    print(f"   threshold {thr}: TP {int((yb&pb).sum())} FN {int((yb&~pb).sum())} FP {int((~yb&pb).sum())} -> recall {(yb&pb).sum()/yb.sum():.3f} precision {(yb&pb).sum()/max(pb.sum(),1):.3f}")
df["pc_phys"]=lpm; df["err"]=em
bad = df[m6 & (np.abs(em)>1)]
print(f"\nDISAGREE |err|>1 dex (per-mission HBR), risk>=-10: {len(bad)} ({len(bad)/m6.sum()*100:.2f}%)")
print("  by object type share bad vs all:", bad.c_object_type.value_counts(normalize=True).round(3).to_dict(), df[m6].c_object_type.value_counts(normalize=True).round(3).to_dict())
print("  MAE by object type:", df[m6].groupby("c_object_type").err.apply(lambda s: np.mean(np.abs(s))).round(3).to_dict())
print("  MAE by mission:", df[m6].groupby("mission_id").err.apply(lambda s: np.mean(np.abs(s))).round(2).to_dict())
for c in ["relative_speed","c_sigma_t","max_risk_scaling","time_to_tca","miss_distance"]:
    print(f"  bad median {c}: {bad[c].median():.2f} vs all {df[m6][c].median():.2f}")
print("  share bad where physics > risk:", round((bad.err>0).mean(),3))
lo = (df.relative_speed<1000).values
print("  rows relative_speed<1000 m/s:", int(lo.sum()), "MAE", round(float(np.nanmean(np.abs(em[lo&m6]))),3))
cols = ["event_id","mission_id","time_to_tca","risk","pc_phys","miss_distance","relative_speed","c_object_type","c_sigma_r","c_sigma_t","c_sigma_n"]
print(bad.sort_values("err").head(4)[cols].round(2).to_string()); print(bad.sort_values("err").tail(4)[cols].round(2).to_string())
# crude 13-col
print("\n--- CRUDE (13 allowed cols: miss + combined sigma_r, isotropic 2D) ---")
for hbr in [5,10,20]:
    lc = pc_crude_log10(df.miss_distance.values, df.t_sigma_r.values, df.c_sigma_r.values, hbr); ec = lc-y
    print(f"HBR={hbr}: pearson(risk>-30, floored -30)={pearsonr(np.maximum(lc,-30)[m_all],y[m_all])[0]:.4f} spearman={spearmanr(lc[m_all],y[m_all])[0]:.4f} MAE(risk>=-10)={np.mean(np.abs(ec[m6])):.1f} median err={np.median(ec[m6]):+.1f} agree(-30 floor)={np.mean((lc<=-30)==(y<=-30)):.4f}")
lc = pc_crude_log10(df.miss_distance.values, df.t_sigma_r.values, df.c_sigma_r.values, hbr_star); yb=y>=-6; pb=lc>=-6
print("   crude threshold -6: TP", int((yb&pb).sum()), "FN", int((yb&~pb).sum()), "FP", int((~yb&pb).sum()))
dd = df.miss_distance.values; lmax = np.log10(hbr_star**2/(np.e*dd**2))
print("   crude dilution-max Pc = HBR^2/(e d^2): pearson with max_risk_estimate", round(pearsonr(lmax, df.max_risk_estimate.values)[0],4), " MAE", round(np.mean(np.abs(lmax-df.max_risk_estimate.values)),3))
print("   fraction risk <= max_risk_estimate:", round((raw.risk <= raw.max_risk_estimate+1e-6).mean(),4))
df[["event_id","mission_id","time_to_tca","risk","pc_phys","err"]].to_parquet(CACHE+"/pc_phys.parquet")
