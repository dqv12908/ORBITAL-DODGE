import numpy as np
from scipy.special import logsumexp
MU = 398600.4418e9  # m^3/s^2

def cov_rtn(s_r, s_t, s_n, c_tr, c_nr, c_nt):
    n = len(s_r); C = np.zeros((n,3,3))
    C[:,0,0]=s_r**2; C[:,1,1]=s_t**2; C[:,2,2]=s_n**2
    C[:,0,1]=C[:,1,0]=c_tr*s_r*s_t; C[:,0,2]=C[:,2,0]=c_nr*s_r*s_n; C[:,1,2]=C[:,2,1]=c_nt*s_t*s_n
    return C

def chaser_rot(df):
    """rotation taking chaser-RTN vectors into target-RTN (approx: same radial, target v from vis-viva on sma)"""
    a = df.t_j2k_sma.values*1e3; r = a  # near-circular LEO (ecc<=0.021)
    vt = np.sqrt(MU*(2/r - 1/a))
    vrel = df[["relative_velocity_r","relative_velocity_t","relative_velocity_n"]].values
    vc = np.stack([np.zeros_like(vt), vt, np.zeros_like(vt)],1) + vrel
    R = np.tile([1.,0,0], (len(df),1))
    N = np.cross(R, vc); N /= np.linalg.norm(N,axis=1,keepdims=True)
    T = np.cross(N, R)
    return np.stack([R,T,N],2)   # columns = chaser axes in target frame -> v_target = M @ v_chaser

def pc2d_log10(df, hbr, rotate=True, nr=24, nth=48, chunk=20000):
    Ct = cov_rtn(df.t_sigma_r.values, df.t_sigma_t.values, df.t_sigma_n.values, df.t_ct_r.values, df.t_cn_r.values, df.t_cn_t.values)
    Cc = cov_rtn(df.c_sigma_r.values, df.c_sigma_t.values, df.c_sigma_n.values, df.c_ct_r.values, df.c_cn_r.values, df.c_cn_t.values)
    if rotate:
        M = chaser_rot(df); Cc = M @ Cc @ np.transpose(M,(0,2,1))
    C = Ct + Cc
    rp = df[["relative_position_r","relative_position_t","relative_position_n"]].values
    v = df[["relative_velocity_r","relative_velocity_t","relative_velocity_n"]].values
    vh = v/np.linalg.norm(v,axis=1,keepdims=True)
    # encounter plane basis: x along miss vector (made perpendicular to v), y = vh x x
    x = rp - (rp*vh).sum(1,keepdims=True)*vh; d = np.linalg.norm(x,axis=1); x /= d[:,None]
    yv = np.cross(vh, x)
    B = np.stack([x,yv],1)  # (n,2,3)
    C2 = B @ C @ np.transpose(B,(0,2,1))
    out = np.full(len(df), np.nan)
    # Gauss-Legendre in r (0..hbr), uniform in theta
    gx, gw = np.polynomial.legendre.leggauss(nr)
    rr = (gx+1)/2*hbr; wr = gw/2*hbr
    th = np.linspace(0,2*np.pi,nth,endpoint=False); wth = 2*np.pi/nth
    px = (rr[:,None]*np.cos(th)[None,:]).ravel(); py = (rr[:,None]*np.sin(th)[None,:]).ravel()
    lw = np.log((wr[:,None]*rr[:,None]*wth*np.ones(nth)[None,:]).ravel())
    for s in range(0,len(df),chunk):
        sl = slice(s,s+chunk)
        c = C2[sl]; det = c[:,0,0]*c[:,1,1]-c[:,0,1]**2
        ia, ib, ic = c[:,1,1]/det, -c[:,0,1]/det, c[:,0,0]/det
        dx = px[None,:]-d[sl,None]; dy = py[None,:]
        q = ia[:,None]*dx*dx + 2*ib[:,None]*dx*dy + ic[:,None]*dy*dy
        lp = -0.5*q - np.log(2*np.pi*np.sqrt(det))[:,None] + lw[None,:]
        out[sl] = logsumexp(lp,axis=1)/np.log(10)
    return out, d, C2

def pc_crude_log10(miss, sig_r_t, sig_r_c, hbr):
    """13-column version: isotropic 2D Gaussian, sigma = combined radial sigma; small-HBR approximation"""
    s2 = sig_r_t**2 + sig_r_c**2
    return (np.log(hbr**2/(2*s2)) - miss**2/(2*s2))/np.log(10)
