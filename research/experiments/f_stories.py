import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
tr = load("train").sort_values(["event_id","time_to_tca"], ascending=[True,False])
print("elevation vs -asin(vr/|v|): MAE", round(float(np.mean(np.abs(tr.elevation + np.degrees(np.arcsin(tr.relative_velocity_r/tr.relative_speed))))),3))
g = tr.groupby("event_id")
E = pd.DataFrame({"n": g.size(), "mission": g.mission_id.first(), "otype": g.c_object_type.last(),
    "final_ttca": g.time_to_tca.last(), "final_risk": g.risk.last(), "final_miss": g.miss_distance.last(),
    "first_ttca": g.time_to_tca.first()})
early = tr[tr.time_to_tca>=2].groupby("event_id")
E["early_max_risk"] = early.risk.max(); E["early_last_risk"] = early.risk.last(); E["early_min_miss"] = early.miss_distance.min()
E["early_last_miss"] = early.miss_distance.last()
E["max_AP"] = g.AP.max(); E["max_F10"] = g.F10.max()
E["sentinel_rows"] = tr.assign(s=((tr.t_sigma_r>6e7)|(tr.c_sigma_r>6e7))).groupby("event_id").s.sum()
E["max_c_sigma_t"] = tr.assign(x=tr.c_sigma_t.where(tr.c_sigma_t<1e7)).groupby("event_id").x.max()
E["final_tier"] = tier(E.final_risk, E.final_miss); E["early_tier"] = tier(E.early_last_risk.fillna(-30), E.early_last_miss.fillna(1e9))
def traj(eid, k=25):
    x = tr[tr.event_id==eid]
    s = " ".join(f"({t:.2f}d,{r:.2f},{m/1000:.2f}km)" for t,r,m in zip(x.time_to_tca, x.risk, x.miss_distance))
    h = E.loc[eid]
    return f"event {eid} | mission {h.mission} | {h.otype} | n={h.n} | final tier {['safe','elevated','critical'][h.final_tier]}\n    (ttca, risk, miss): {s}"
el = E[(E.final_ttca<1)&(E.first_ttca>=2)]
print("\n(1) LATE ESCALATION: early (>=2d) all risk<=-8 and tier safe/elevated, final risk>=-5")
c1 = el[(el.early_max_risk<=-8)&(el.final_risk>=-5)].assign(jump=lambda d: d.final_risk-d.early_max_risk).sort_values("jump", ascending=False)
print("   candidates:", len(c1), c1.head(8)[["mission","otype","early_max_risk","final_risk","final_miss","jump"]].round(2).to_string())
for e in c1.index[:3]: print(traj(e))
print("\n   late escalation by MISS (early tier safe, final miss<1 km):")
c1b = el[(el.early_tier==0)&(el.final_miss<1000)].sort_values("final_miss")
print("   candidates:", len(c1b)); 
for e in c1b.index[:2]: print(traj(e))
print("\n(2) SCARY EARLY -> -30: early max risk>=-5, final risk -30")
c2 = el[(el.early_max_risk>=-5)&(el.final_risk<=-30)].sort_values("early_max_risk", ascending=False)
print("   candidates:", len(c2), c2.head(8)[["mission","otype","early_max_risk","final_risk","final_miss"]].round(2).to_string())
for e in c2.index[:3]: print(traj(e))
print("\n(3) MISS<1km BUT TINY Pc at final (risk<=-10): tier rule says critical, probability says negligible")
c3 = E[(E.final_miss<1000)&(E.final_risk<=-10)].sort_values("final_miss")
print("   count:", len(c3), " of all final miss<1km:", int((E.final_miss<1000).sum()), " final miss<1km & risk<-6:", int(((E.final_miss<1000)&(E.final_risk<-6)).sum()), " & risk=-30:", int(((E.final_miss<1000)&(E.final_risk<=-30)).sum()))
for e in c3.index[:3]: print(traj(e))
print("\n   converse: Pc>=1e-4 at final (risk>=-4) list:", E[E.final_risk>=-4].sort_values("final_risk",ascending=False)[["mission","otype","final_risk","final_miss","final_ttca"]].round(3).to_string())
print("\n(4) SENTINEL / HUGE COVARIANCE")
c4 = E[(E.sentinel_rows>0)&(E.sentinel_rows<E.n)].sort_values("sentinel_rows", ascending=False)
print("   events with some (not all) sentinel rows:", len(c4), "; all rows sentinel:", int((E.sentinel_rows==E.n).sum()))
for e in c4.index[:2]:
    x = tr[tr.event_id==e]; print(traj(e)); print("    sigma_r t/c:", [(round(a,1) if a<6e7 else "SENT", round(b,1) if b<6e7 else "SENT") for a,b in zip(x.t_sigma_r,x.c_sigma_r)])
c4b = E.sort_values("max_c_sigma_t", ascending=False)
print("   largest non-sentinel c_sigma_t events:", c4b.head(3)[["max_c_sigma_t","final_risk","final_miss","otype"]].round(1).to_string())
print("\n(5) HIGH SPACE WEATHER")
print("   AP quantiles (rows):", tr.AP.quantile([.5,.9,.99,1]).to_dict(), " F10:", tr.F10.quantile([.5,.9,.99,1]).to_dict())
c5 = E[(E.max_AP>=60)].sort_values(["final_risk"], ascending=False)
print("   events with AP>=60:", len(c5), c5.head(6)[["mission","otype","max_AP","max_F10","final_risk","final_miss"]].round(2).to_string())
for e in c5.index[:2]:
    x = tr[tr.event_id==e]; print(traj(e)); print("    AP:", x.AP.tolist(), " F10:", x.F10.tolist())
c5b = E[(E.max_F10>=200)].sort_values("final_risk", ascending=False)
print("   events with F10>=200:", len(c5b), c5b.head(3)[["mission","max_F10","max_AP","final_risk","final_miss"]].round(2).to_string())
# does space weather matter? risk change vs AP
el2 = el.dropna(subset=["early_last_risk"]).copy(); el2["d"] = el2.final_risk.clip(-10) - el2.early_last_risk.clip(-10)
print("   corr(max_AP, |final-early risk| clipped -10):", round(float(np.corrcoef(el2.max_AP.fillna(el2.max_AP.median()), el2.d.abs())[0,1]),3),
      "; median |change| AP>=30:", round(float(el2[el2.max_AP>=30].d.abs().median()),3), "vs AP<30:", round(float(el2[el2.max_AP<30].d.abs().median()),3))
print("\n(6) PHYSICS DISAGREEMENT example (mission 6, risk ~-5 but 2D-Pc ~-480): event 4200")
print(traj(4200))
print("\n(7) post-TCA final CDM (ttca<0) count:", int((E.final_ttca<0).sum()))
E.to_parquet(CACHE+"/event_summary.parquet")
