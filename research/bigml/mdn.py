"""Mixture-density transition model CDM_k -> CDM_{k+1} and GPU Monte Carlo rollouts to TCA."""
import math, time, numpy as np, torch, torch.nn as nn
from bm import XCOLS, YD, LS

IX = {c: i for i, c in enumerate(XCOLS)}


class MDN(nn.Module):
    def __init__(self, din, K=10, H=256, D=YD):
        super().__init__()
        self.K, self.D = K, D
        self.body = nn.Sequential(nn.Linear(din, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(),
                                  nn.Linear(H, H), nn.SiLU())
        self.head = nn.Linear(H, K * (1 + 2 * D + 1))
        self.term = nn.Sequential(nn.Linear(din, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.head(self.body(x))
        K, D = self.K, self.D
        logit = h[:, :K]
        mu = h[:, K:K + K * D].view(-1, K, D)
        ls = h[:, K + K * D:K + 2 * K * D].view(-1, K, D).clamp(-7, 3)
        fl = h[:, K + 2 * K * D:]
        return logit, mu, ls, fl


def nll(model, x, y, f):
    logit, mu, ls, fl = model(x)
    m = ~torch.isnan(y)
    yz = torch.where(m, y, torch.zeros_like(y))[:, None, :]
    lp = -0.5 * ((yz - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
    lp = (lp * m[:, None, :]).sum(-1)
    lpf = -nn.functional.binary_cross_entropy_with_logits(fl, f[:, None].expand_as(fl), reduction="none")
    tot = torch.logsumexp(torch.log_softmax(logit, -1) + lp + lpf, -1)
    return -tot.mean()


class Futures:
    """fit / rollout wrapper holding normalisation stats."""

    def __init__(self, dev="cuda", K=10, H=256):
        self.dev, self.K, self.H = dev, K, H

    def fit(self, Xp, Y, F, Xt, T, epochs=int(__import__('os').environ.get('EPOCHS', 40)), bs=4096, lr=2e-3, seed=0, log=print):
        torch.manual_seed(seed); np.random.seed(seed)
        self.xm = np.nanmean(Xp, 0); self.xs = np.nanstd(Xp, 0) + 1e-6
        self.ym = np.nanmean(Y, 0); self.ys = np.nanstd(Y, 0) + 1e-6
        dev = self.dev
        nx = lambda X: torch.tensor(np.nan_to_num((X - self.xm) / self.xs), dtype=torch.float32, device=dev)
        x = nx(Xp); y = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32, device=dev)
        f = torch.tensor(F, dtype=torch.float32, device=dev)
        xt = nx(Xt); tt = torch.tensor(T, dtype=torch.float32, device=dev)
        n = len(x); perm = torch.randperm(n, device=dev); nv = n // 10
        vi, ti = perm[:nv], perm[nv:]
        self.model = MDN(x.shape[1], self.K, self.H).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        steps = epochs * (len(ti) // bs + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        best, bstate = 1e9, None
        t0 = time.time()
        for ep in range(epochs):
            self.model.train()
            p = ti[torch.randperm(len(ti), device=dev)]
            for i in range(0, len(p), bs):
                b = p[i:i + bs]
                loss = nll(self.model, x[b], y[b], f[b])
                # termination head (subsample of rows)
                tb = torch.randint(0, len(xt), (min(bs, len(xt)),), device=dev)
                lt = nn.functional.binary_cross_entropy_with_logits(self.model.term(xt[tb])[:, 0], tt[tb])
                opt.zero_grad(); (loss + lt).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step(); sch.step()
            self.model.eval()
            with torch.no_grad():
                v = float(np.mean([nll(self.model, x[vi[j:j + 16384]], y[vi[j:j + 16384]], f[vi[j:j + 16384]]).item()
                                   for j in range(0, nv, 16384)]))
            if v < best:
                best = v; bstate = {k: t.clone() for k, t in self.model.state_dict().items()}
            if ep % 5 == 0 or ep == epochs - 1:
                log(f"  ep {ep} train {loss.item():.3f} val {v:.3f} term {lt.item():.3f} ({time.time()-t0:.0f}s)")
        self.model.load_state_dict(bstate); self.model.eval()
        self.val_nll = best
        return self

    def state_dict(self):
        return dict(model=self.model.state_dict(), xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys, K=self.K, H=self.H)

    @torch.no_grad()
    def rollout(self, X0, n_paths, grid, max_steps=30, seed=0, keep_paths=0):
        """X0: (E, din) raw start states. Returns dict with p_tiers (E,3), q_rk/q_lm (E,5,G),
        final rk/lm samples summary, and optionally kept paths."""
        dev = self.dev; g = torch.Generator(device=dev); g.manual_seed(seed)
        E = len(X0); P = n_paths
        xm = torch.tensor(self.xm, dtype=torch.float32, device=dev); xs = torch.tensor(self.xs, dtype=torch.float32, device=dev)
        ym = torch.tensor(self.ym, dtype=torch.float32, device=dev); ys = torch.tensor(self.ys, dtype=torch.float32, device=dev)
        S = torch.tensor(X0, dtype=torch.float32, device=dev).repeat_interleave(P, 0)  # (E*P, din)
        N = len(S)
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        Ts, Ls, Rs = [S[:, IX["time_to_tca"]].clone()], [S[:, IX["lm"]].clone()], [S[:, IX["rk"]].clone()]
        lsi = [IX[c] for c in LS]
        for step in range(max_steps):
            if not alive.any():
                break
            xin = torch.nan_to_num((S - xm) / xs)
            logit, mu, lsg, fl = self.model(xin)
            k = torch.multinomial(torch.softmax(logit, -1), 1, generator=g)[:, 0]
            ar = torch.arange(N, device=dev)
            z = mu[ar, k] + lsg[ar, k].exp() * torch.randn(N, YD, device=dev, generator=g)
            yv = z * ys + ym
            isfl = torch.rand(N, device=dev, generator=g) < torch.sigmoid(fl[ar, k])
            dt = yv[:, 0].exp().clamp(1e-3, 3.0)
            old_t = S[:, IX["time_to_tca"]]; old_lm = S[:, IX["lm"]]; old_rk = S[:, IX["rk"]]
            new_t = old_t - dt
            new_lm = (old_lm + yv[:, 1]).clamp(0.0, 6.5)
            new_rk = torch.where(isfl, torch.full_like(old_rk, -30.0), yv[:, 2].clamp(-29.99, 0.0))
            a = alive
            S2 = S.clone()
            S2[:, IX["time_to_tca"]] = new_t; S2[:, IX["lm"]] = new_lm; S2[:, IX["rk"]] = new_rk
            S2[:, IX["floor"]] = isfl.float()
            for j, c in enumerate(lsi):
                S2[:, c] = S[:, c] + yv[:, 3 + j]  # NaN stays NaN (unknown covariance)
            S2[:, IX["prev_dlm"]] = new_lm - old_lm; S2[:, IX["prev_drk"]] = new_rk - old_rk; S2[:, IX["prev_dt"]] = dt
            S2[:, IX["has_prev"]] = 1.0; S2[:, IX["cdm_index"]] = S[:, IX["cdm_index"]] + 1
            S2[:, IX["rk_max"]] = torch.maximum(S[:, IX["rk_max"]], new_rk)
            S2[:, IX["lm_min"]] = torch.minimum(S[:, IX["lm_min"]], new_lm)
            S = torch.where(a[:, None], S2, S)
            # termination decided on the new state
            pterm = torch.sigmoid(self.model.term(torch.nan_to_num((S - xm) / xs))[:, 0])
            tnow = S[:, IX["time_to_tca"]]
            stop = (tnow < 1.0) & (torch.rand(N, device=dev, generator=g) < pterm)
            stop |= tnow < -0.15
            stop |= (step == max_steps - 1)
            Ts.append(torch.where(a, tnow, torch.full_like(tnow, -1e9)))
            Ls.append(S[:, IX["lm"]].clone()); Rs.append(S[:, IX["rk"]].clone())
            alive = a & ~stop
        while len(Ts) < max_steps + 1:
            Ts.append(torch.full_like(Ts[0], -1e9)); Ls.append(Ls[-1]); Rs.append(Rs[-1])
        T = torch.stack(Ts, 1); L = torch.stack(Ls, 1); R = torch.stack(Rs, 1)  # (N, max_steps+1)
        lmF, rkF = S[:, IX["lm"]], S[:, IX["rk"]]
        crit = (rkF >= -4) | (lmF < 3.0)
        elev = ~crit & (((rkF >= -6) & (rkF < -4)) | ((lmF >= 3.0) & (lmF <= math.log10(5000))))
        tiers = torch.where(crit, 2, torch.where(elev, 1, 0)).view(E, P)
        p_t = torch.stack([(tiers == i).float().mean(1) for i in range(3)], 1)
        gt = torch.tensor(grid, dtype=torch.float32, device=dev)
        cnt = (T[:, :, None] >= gt[None, None, :]).sum(1) - 1  # (N, G)
        cnt = cnt.clamp(min=0)
        Lg = torch.gather(L, 1, cnt).view(E, P, -1); Rg = torch.gather(R, 1, cnt).view(E, P, -1)
        qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=dev)
        q_lm = torch.quantile(Lg, qs, dim=1).permute(1, 0, 2)  # (E,5,G)
        q_rk = torch.quantile(Rg, qs, dim=1).permute(1, 0, 2)
        out = dict(p_tiers=p_t.cpu().numpy(), q_lm=q_lm.cpu().numpy(), q_rk=q_rk.cpu().numpy(),
                   n_steps=(T > -1e8).sum(1).float().view(E, P).mean(1).cpu().numpy(),
                   final_ttca_med=torch.quantile(torch.where(T > -1e8, T, torch.full_like(T, 1e9)).min(1).values.view(E, P), 0.5, dim=1).cpu().numpy())
        if keep_paths:
            kp = min(keep_paths, P)
            sel = (torch.arange(E, device=dev)[:, None] * P + torch.arange(kp, device=dev)[None, :]).reshape(-1)
            out["paths"] = dict(T=T[sel].cpu().numpy().astype(np.float32), L=L[sel].cpu().numpy().astype(np.float16),
                                R=R[sel].cpu().numpy().astype(np.float16))
        return out
