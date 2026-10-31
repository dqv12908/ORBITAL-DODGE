import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
tr = load("train").sort_values(["event_id","time_to_tca"], ascending=[True,False])
g = tr.groupby("event_id")
n = g.size()
print("CDMs/event quantiles:", n.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(), " events with 1 CDM:", int((n==1).sum()))
first = g.time_to_tca.first(); last = g.time_to_tca.last()
print("first CDM ttca quantiles (d):", first.quantile([.05,.25,.5,.75,.95]).round(2).to_dict())
print("last CDM ttca quantiles (d):", last.quantile([.05,.25,.5,.75,.95]).round(2).to_dict())
print("events whose last CDM < 1 d:", int((last<1).sum()), " last CDM < 0 (post-TCA):", int((last<0).sum()))
gap = -g.time_to_tca.diff().dropna()*24
print("CDM spacing (h) quantiles:", gap.quantile([.05,.1,.25,.5,.75,.9,.95,.99]).round(2).to_dict(), " mean", round(gap.mean(),2))
print("  share of gaps < 1h:", round((gap<1).mean(),3), " 6-10 h:", round(((gap>=6)&(gap<=10)).mean(),3), " 10-14h:", round(((gap>10)&(gap<=14)).mean(),3), " >24h:", round((gap>24).mean(),3))
hist = np.histogram(gap, bins=[0,1,2,4,6,8,10,12,16,20,24,48,200])
print("  histogram:", dict(zip([f"{a}-{b}h" for a,b in zip(hist[1][:-1],hist[1][1:])], hist[0].tolist())))
print("  spacing by ttca band (median h):", {b: round(float(gap[(tr.loc[gap.index].time_to_tca>=lo)&(tr.loc[gap.index].time_to_tca<hi)].median()),2) for b,(lo,hi) in {"<1d":(-1,1),"1-2d":(1,2),"2-4d":(2,4),"4-7d":(4,7)}.items()})
# CDM release time-of-day proxy: frac part of ttca*24? (TCA times are random, so no)
epm = tr.groupby("mission_id").event_id.nunique().sort_values(ascending=False)
print("events per mission:", epm.to_dict())
print("CDMs per mission:", tr.groupby("mission_id").size().sort_values(ascending=False).to_dict())
# 24h window from time_to_tca: rows in the [0,1) d band
in24 = tr[(tr.time_to_tca>=0)&(tr.time_to_tca<1)]
print("CDMs with 0<=ttca<1:", len(in24), " events:", in24.event_id.nunique())
lastc = tr.groupby("event_id").tail(1)
dang = lastc[(lastc.time_to_tca<1)]
t = tier(dang.risk, dang.miss_distance)
print("events with final CDM inside the last 24 h, by final tier:", dict(zip(["safe","elev","crit"], np.bincount(t, minlength=3).tolist())))
print("  of these miss<1km:", int((dang.miss_distance<1000).sum()), " miss<5km:", int((dang.miss_distance<5000).sum()))
# space-weather fingerprint: do (F10,F3M,SSN,AP) tuples behave like calendar days?
sw = tr.dropna(subset=["F10","F3M","SSN","AP"])
tup = sw[["F10","F3M","SSN","AP"]].round(3).astype(str).agg("|".join, axis=1)
print("distinct (F10,F3M,SSN,AP) tuples:", tup.nunique(), " distinct (F10,F3M,SSN):", sw[["F10","F3M","SSN"]].drop_duplicates().shape[0], " distinct F3M:", sw.F3M.nunique())
sw = sw.assign(tup=tup.values)
per_ev = sw.groupby("event_id").tup.nunique()
print("distinct SW tuples per event quantiles:", per_ev.quantile([.1,.5,.9]).to_dict(), " vs span of event (d) median", round(float((first-last).median()),2))
# within an event, tuple changes should happen ~once per day
def changes(x):
    t = x.tup.values; return (t[1:]!=t[:-1]).sum()
ch = sw.groupby("event_id")[["tup"]].apply(changes)
span = (first-last).reindex(ch.index)
print("tuple changes per event-day (median):", round(float((ch/span.clip(lower=0.5)).median()),3))
# co-occurrence: a tuple shared by how many events/missions -> 'same calendar day'
te = sw.groupby("tup").agg(events=("event_id","nunique"), missions=("mission_id","nunique"))
print("events per SW-tuple (='day'):", te.events.quantile([.5,.9,.99]).to_dict(), " max", int(te.events.max()), "; missions per tuple median", te.missions.median())
print("F10 range", sw.F10.min(), sw.F10.max(), "AP range", sw.AP.min(), sw.AP.max(), " SSN range", sw.SSN.min(), sw.SSN.max())
