"""Reconstruct a relative calendar from space-weather fingerprints: each (F10,F3M,SSN,AP) tuple ~ one UTC day.
For every pair of CDMs in an event: day(tupB)-day(tupA) ~= ttcaA-ttcaB (creation-time difference, days). Solve by sparse least squares."""
import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr
tr = pd.concat([load("train").assign(split="train"), load("test").assign(split="test")])
tr = tr.dropna(subset=["F10","F3M","SSN","AP"]).copy()
tr["tup"] = tr[["F10","F3M","SSN"]].round(3).astype(str).agg("|".join, axis=1)
codes, uniq = pd.factorize(tr.tup); tr["day"] = codes; K = len(uniq)
tr["key"] = tr.split + ":" + tr.event_id.astype(str)
tr = tr.sort_values(["key","time_to_tca"], ascending=[True,False])
a = tr.groupby("key").day.shift(); ta = tr.groupby("key").time_to_tca.shift()
p = tr[a.notna()].assign(a=a.dropna().astype(int), ta=ta.dropna())
p = p[p.day != p.a]
rows = np.arange(len(p)); dt = (p.ta - p.time_to_tca).values
adj = coo_matrix((np.ones(len(p)), (p.a.values, p.day.values)), shape=(K,K))
nc, lab = connected_components(adj, directed=False)
sizes = np.bincount(lab); big = sizes.argmax()
print(f"tuples(days) K={K}, day-change pairs={len(p)}, connected components={nc}, largest component={sizes.max()} tuples, singletons={int((sizes==1).sum())}")
sel = (lab[p.a.values]==big)
A = coo_matrix((np.r_[-np.ones(sel.sum()), np.ones(sel.sum())], (np.r_[np.arange(sel.sum()),np.arange(sel.sum())], np.r_[p.a.values[sel], p.day.values[sel]])), shape=(sel.sum(),K)).tocsr()
from scipy.sparse.linalg import spsolve; from scipy.sparse import vstack, csr_matrix; Aa = vstack([A, csr_matrix(([1.0],([0],[0])),shape=(1,K))]).tocsc(); x = spsolve((Aa.T@Aa).tocsc(), Aa.T@np.r_[dt[sel],0.0])
res = A@x - dt[sel]
print(f"lsqr residual (days): median |r|={np.median(np.abs(res)):.3f}, p90={np.quantile(np.abs(res),.9):.3f}, RMS={np.sqrt(np.mean(res**2)):.3f}")
xs = x[lab==big]; xs = xs - xs.min()
print(f"reconstructed calendar span (largest component): {xs.max():.0f} days (~{xs.max()/365.25:.2f} years)")
# integer consistency: fractional parts of solved day positions should cluster if each tuple = one day
r = np.round(xs); print("share of tuple positions within 0.3 d of an integer after rounding:", round(float(np.mean(np.abs(xs-r)<0.3)),3), "; distinct rounded days:", len(np.unique(r)), "for", len(xs), "tuples")
# F10 smoothness along reconstructed time order: F3M (81-day mean) should be smooth
u = pd.DataFrame([s.split("|") for s in uniq], columns=["F10","F3M","SSN"]).astype(float)
u = u[lab==big].assign(t=xs).sort_values("t")
print("lag-1 autocorr along reconstructed order: F3M", round(u.F3M.autocorr(),3), " F10", round(u.F10.autocorr(),3), " SSN", round(u.SSN.autocorr(),3))
print("random-order control lag-1 autocorr F3M", round(u.F3M.sample(frac=1, random_state=0).reset_index(drop=True).autocorr(),3))
print("first/last reconstructed days F3M:", u.F3M.head(3).tolist(), u.F3M.tail(3).tolist())
u.to_csv(CACHE+"/sw_calendar.csv", index=False)
