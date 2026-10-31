import sys; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
from common import *
tr = load("train"); te = load("test")
print("train", tr.shape, "events", tr.event_id.nunique(), "test", te.shape)
inp, tgt = kelvins_frame(tr)
print("eligible events", len(tgt))
latest = inp.sort_values(["event_id","time_to_tca"]).groupby("event_id").head(1).set_index("event_id").loc[tgt.index]
print("latest >=2d ttca describe", latest.time_to_tca.describe().round(3).to_dict())
print("final ttca describe", tgt.y_ttca.describe().round(3).to_dict())
hr = tgt.y_risk >= -6
print("(a) high risk final>=-6:", hr.sum(), f"{hr.mean()*100:.2f}%")
print("    final risk == -30:", (tgt.y_risk<=-30).sum(), f"{(tgt.y_risk<=-30).mean()*100:.1f}%")
print("    latest >=2d risk>=-6:", (latest.risk>=-6).sum())
yt = tier(tgt.y_risk, tgt.y_miss); xt = tier(latest.risk, latest.miss_distance)
names = ["safe","elevated","critical"]
print("(b) final tiers:", {names[k]: int((yt==k).sum()) for k in range(3)}, {names[k]: round(float((yt==k).mean()*100),2) for k in range(3)})
print("    latest(>=2d) tiers:", {names[k]: int((xt==k).sum()) for k in range(3)})
# critical driver
crit = yt==2
print("    critical by risk>=-4:", int(((tgt.y_risk>=-4)&crit).sum()), " by miss<1km:", int(((tgt.y_miss<1000)&crit).sum()), " both:", int(((tgt.y_risk>=-4)&(tgt.y_miss<1000)).sum()))
elev = yt==1
print("    elevated by risk in[-6,-4):", int(((tgt.y_risk>=-6)&(tgt.y_risk<-4)&elev).sum()), " by miss 1-5km:", int(((tgt.y_miss>=1000)&(tgt.y_miss<=5000)&elev).sum()))
ct = pd.crosstab(pd.Series(xt, name="latest>=2d"), pd.Series(yt, name="final"))
ct.index=[names[i] for i in ct.index]; ct.columns=[names[i] for i in ct.columns]
print("transition counts\n", ct)
print("row %\n", (ct.div(ct.sum(1), axis=0)*100).round(1))
print("tier changed:", int((xt!=yt).sum()), f"{(xt!=yt).mean()*100:.1f}%", " up:", int((yt>xt).sum()), " down:", int((yt<xt).sum()))
# high-risk transition
ct2 = pd.crosstab(latest.risk>=-6, tgt.y_risk>=-6); print("risk>=-6 latest vs final\n", ct2)
# miss distance change
dm = (tgt.y_miss - latest.miss_distance)
print("miss distance change final-latest (m) quantiles", dm.quantile([.05,.25,.5,.75,.95]).round(0).to_dict())
print("|log10 miss ratio| median", np.median(np.abs(np.log10(tgt.y_miss.clip(1)/latest.miss_distance.clip(1)))).round(3))
# whole-train tiers (all events final CDM)
last = tr.sort_values(["event_id","time_to_tca"]).groupby("event_id").head(1)
at = tier(last.risk,last.miss_distance); print("all-train final tiers", np.bincount(at))
# test
tl = te.sort_values(["event_id","time_to_tca"]).groupby("event_id").head(1)
tt = tier(tl.risk, tl.miss_distance)
print("test latest-CDM tiers", np.bincount(tt), " test latest risk>=-6:", int((tl.risk>=-6).sum()), " test latest ttca range", tl.time_to_tca.min(), tl.time_to_tca.max())
print("test missions", sorted(te.mission_id.unique()), " train missions", sorted(tr.mission_id.unique()))
