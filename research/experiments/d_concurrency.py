"""How many events are 'live' on the same day? Use the daily space-weather fingerprint (F10,F3M,SSN) as a day key (approximate)."""
import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
tr = load("train").dropna(subset=["F10","F3M","SSN"])
tr["day"] = tr[["F10","F3M","SSN"]].round(3).astype(str).agg("|".join, axis=1)
a = tr.groupby(["mission_id","day"]).event_id.nunique()
print("events with >=1 CDM sharing a day-key, per mission-day: quantiles", a.quantile([.25,.5,.75,.9,.99]).to_dict(), "max", int(a.max()))
print("per mission median:", a.groupby(level=0).median().to_dict())
c = tr.groupby(["mission_id","day"]).size()
print("CDMs per mission-day quantiles:", c.quantile([.25,.5,.75,.9,.99]).to_dict())
# all missions combined (a 'fleet operator' view)
b = tr.groupby("day").event_id.nunique(); print("all-mission events per day-key: median", b.median(), "p90", b.quantile(.9), "max", b.max())
crit_day = tr[(tr.miss_distance<1000)|(tr.risk>=-4)].groupby("day").event_id.nunique().reindex(b.index).fillna(0)
print("critical-looking events (any CDM miss<1km or risk>=-4) per day-key: median", crit_day.median(), "p90", crit_day.quantile(.9), "max", crit_day.max())
