import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
tr = load("train").sort_values(["event_id","time_to_tca"], ascending=[True,False]); E = pd.read_parquet(CACHE+"/event_summary.parquet")
def traj(eid):
    x = tr[tr.event_id==eid]; h = E.loc[eid]
    return f"event {eid} | mission {h.mission} | {h.otype} | n={h.n}\n    " + " ".join(f"({t:.2f}d,{r:.2f},{m/1000:.2f}km)" for t,r,m in zip(x.time_to_tca, x.risk, x.miss_distance))
el = E[(E.final_ttca<1)&(E.first_ttca>=2)].assign(jump=lambda d: d.final_risk-d.early_max_risk)
c = el[(el.final_risk>=-5.5)].sort_values("jump", ascending=False)
print("(1) largest late risk increases among Kelvins-eligible events (final risk>=-5.5):")
print(c.head(10)[["mission","otype","early_max_risk","early_last_risk","final_risk","final_miss","jump"]].round(2).to_string())
for e in c.index[:3]: print(traj(e))
for e in [8767, 8863, 3555]: print(traj(e))
# test-set stories: highest latest risk in test
te = load("test").sort_values(["event_id","time_to_tca"], ascending=[True,False])
tl = te.groupby("event_id").tail(1)
print("\nTEST (no ground truth): top latest-risk events:", tl.sort_values("risk", ascending=False).head(5)[["event_id","mission_id","time_to_tca","risk","miss_distance","c_object_type"]].round(3).to_string(index=False))
