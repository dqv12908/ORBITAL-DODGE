"""E. Orbit reconstruction feasibility: place target on a Keplerian orbit through a conjunction point at geocentric_latitude,
derive chaser state = target state + relative state (RTN->ECI), and check the derived chaser elements against c_j2k_sma/ecc/inc."""
import sys, warnings; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); warnings.filterwarnings("ignore")
from common import *
MU = 398600.4418; RE = 6378.137
tr = load("train").sort_values(["event_id","time_to_tca"])
f = tr.groupby("event_id").head(1).copy()   # final CDM per event
print("final CDMs:", len(f))
# h_apo/h_per consistency with sma/ecc
for s in ["t","c"]:
    ha = f[f"{s}_j2k_sma"]*(1+f[f"{s}_j2k_ecc"])-RE; hp = f[f"{s}_j2k_sma"]*(1-f[f"{s}_j2k_ecc"])-RE
    print(f"{s}: |h_apo - a(1+e)+Re| median {np.median(np.abs(ha-f[f'{s}_h_apo'])):.3f} km p99 {np.quantile(np.abs(ha-f[f'{s}_h_apo']),.99):.3f}; |h_per-..| median {np.median(np.abs(hp-f[f'{s}_h_per'])):.3f} km")
print("chaser sma>8000 km (non-LEO):", int((f.c_j2k_sma>8000).sum()), " chaser ecc>0.1:", int((f.c_j2k_ecc>0.1).sum()), " chaser ecc quantiles", f.c_j2k_ecc.quantile([.5,.9,.99]).round(4).to_dict())
# altitude band overlap test: target and chaser radial bands must overlap at a conjunction
ov = (np.minimum(f.t_h_apo, f.c_h_apo) - np.maximum(f.t_h_per, f.c_h_per))
print("altitude-band overlap >= -5 km (physically consistent):", round(float((ov>=-5).mean()),4), " overlap < -5 km:", int((ov<-5).sum()))
# latitude reachable by target inclination
it = np.radians(f.t_j2k_inc); ic = np.radians(f.c_j2k_inc); phi = np.radians(f.geocentric_latitude)
lim_t = np.degrees(np.minimum(it, np.pi-it)); lim_c = np.degrees(np.minimum(ic, np.pi-ic))
print("|lat| <= target inc limit (+0.5deg):", round(float((np.abs(f.geocentric_latitude)<=lim_t+0.5).mean()),4), "; <= chaser inc limit:", round(float((np.abs(f.geocentric_latitude)<=lim_c+0.5).mean()),4))
# meaning of azimuth/elevation: compare with relative velocity direction
vr, vt, vn = f.relative_velocity_r, f.relative_velocity_t, f.relative_velocity_n
rr, rt, rn = f.relative_position_r, f.relative_position_t, f.relative_position_n
cands = {"atan2(vn,-vt)": np.degrees(np.arctan2(vn,-vt)), "atan2(vn,vt)": np.degrees(np.arctan2(vn,vt)),
         "asin(vr/|v|)": np.degrees(np.arcsin(vr/f.relative_speed)), "asin(rr/|r|)": np.degrees(np.arcsin(rr/f.miss_distance)),
         "atan2(rn,rt)": np.degrees(np.arctan2(rn,rt)), "atan2(rt,rn)": np.degrees(np.arctan2(rt,rn))}
for k,v in cands.items():
    print(f"  {k:16s} corr w/ azimuth {np.corrcoef(v,f.azimuth)[0,1]:+.3f} MAE {np.mean(np.abs(v-f.azimuth)):.2f} | corr w/ elevation {np.corrcoef(v,f.elevation)[0,1]:+.3f} MAE {np.mean(np.abs(v-f.elevation)):.2f}")
# reconstruction
def recon(row_phi, a_t, i_t, branch, vrel, rrel):
    n = len(a_t); r = a_t                       # circular approx at conjunction (ecc<=0.021)
    rhat = np.stack([np.cos(row_phi), np.zeros(n), np.sin(row_phi)],1)   # longitude 0 (free, synthetic)
    east = np.tile([0.,1.,0.],(n,1)); north = np.stack([-np.sin(row_phi), np.zeros(n), np.cos(row_phi)],1)
    s = np.clip(np.cos(i_t)/np.cos(row_phi), -1, 1); psi = np.arcsin(s)          # heading from north
    if branch == "desc": psi = np.pi - psi
    vmag = np.sqrt(MU*(2/r-1/a_t))
    v = vmag[:,None]*(np.cos(psi)[:,None]*north + np.sin(psi)[:,None]*east)
    R = rhat; N = np.cross(rhat, v); N /= np.linalg.norm(N,axis=1,keepdims=True); T = np.cross(N,R)
    dv = (vrel[:,0:1]*R + vrel[:,1:2]*T + vrel[:,2:3]*N)/1000.0   # m/s -> km/s
    dr = (rrel[:,0:1]*R + rrel[:,1:2]*T + rrel[:,2:3]*N)/1000.0
    rc = r[:,None]*rhat + dr; vc = v + dv
    rn_ = np.linalg.norm(rc,axis=1); vn_ = np.linalg.norm(vc,axis=1)
    a_c = 1/(2/rn_ - vn_**2/MU)
    h = np.cross(rc, vc); inc_c = np.degrees(np.arccos(h[:,2]/np.linalg.norm(h,axis=1)))
    evec = np.cross(vc,h)/MU - rc/rn_[:,None]; e_c = np.linalg.norm(evec,axis=1)
    # check target inc reproduced
    ht = np.cross(r[:,None]*rhat, v); inc_t = np.degrees(np.arccos(ht[:,2]/np.linalg.norm(ht,axis=1)))
    return a_c, inc_c, e_c, inc_t
vrel = f[["relative_velocity_r","relative_velocity_t","relative_velocity_n"]].values
rrel = f[["relative_position_r","relative_position_t","relative_position_n"]].values
out = {}
for b in ["asc","desc"]:
    out[b] = recon(phi.values, f.t_j2k_sma.values, it.values, b, vrel, rrel)
print("target inc reproduced (|d|<0.01deg):", round(float(np.mean(np.abs(out['asc'][3]-f.t_j2k_inc.values)<0.01)),4))
di = {b: np.abs(out[b][1]-f.c_j2k_inc.values) for b in out}; da = {b: np.abs(out[b][0]-f.c_j2k_sma.values) for b in out}
best = np.where(di["asc"]+da["asc"]/10 <= di["desc"]+da["desc"]/10, "asc", "desc")
dib = np.where(best=="asc", di["asc"], di["desc"]); dab = np.where(best=="asc", da["asc"], da["desc"])
ec = np.where(best=="asc", out["asc"][2], out["desc"][2])
amb = np.abs(di["asc"]-di["desc"]) < 1.0
print(f"best-branch chaser inc error: median {np.median(dib):.3f} deg, p90 {np.quantile(dib,.9):.3f}, share <1deg {np.mean(dib<1):.3f}, <0.5deg {np.mean(dib<0.5):.3f}")
print(f"best-branch chaser sma error: median {np.median(dab):.2f} km, p90 {np.quantile(dab,.9):.2f}, share <25km {np.mean(dab<25):.3f}, <50 km {np.mean(dab<50):.3f}")
print(f"chaser ecc error median {np.median(np.abs(ec-f.c_j2k_ecc.values)):.4f}")
print(f"branch ambiguity (both branches within 1 deg on inc): {np.mean(amb):.3f}; branch split asc/desc: {np.mean(best=='asc'):.3f}/{np.mean(best=='desc'):.3f}")
print(f"worst-branch chaser inc error median {np.median(np.maximum(di['asc'],di['desc'])):.2f} deg (shows branch choice matters)")
f["inc_err"]=dib; f["sma_err"]=dab
print("errors by chaser sma>8000:", f[f.c_j2k_sma>8000][["inc_err","sma_err"]].median().round(2).to_dict())
lowv = f.relative_speed<1000
print("low relative speed (<1 km/s) events:", int(lowv.sum()), "inc err median", round(float(f[lowv].inc_err.median()),3))
