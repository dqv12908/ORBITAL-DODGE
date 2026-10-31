// Inspector adapters: same call shapes as the old canvas charts (EncounterPlane.set / RiskChart.set), drawn as ggplot cards.
import { PlotCard } from './card.js';
import { api } from '../../net.js';

const r3 = (v) => (v == null || !Number.isFinite(+v) ? v : Math.round(+v * 1000) / 1000);

export class GgEncounter {
  constructor(host) { this.card = new PlotCard(host, { name: 'encounter-plane', base: 11.5 }); }
  /** p: planeParams(); tier; key */
  set(p, tier, key) {
    const q = p ? { sx: r3(p.sx), sy: r3(p.sy), th: r3(p.th), mx: r3(p.mx), my: r3(p.my), hbr: r3(p.hbr), log10: r3(p.log10), status: p.status, k: r3(p.k) } : null;
    this.card.draw('encounterPlane', [q, { key, tier }]);
  }
}

export class GgRisk {
  constructor(host) { this.card = new PlotCard(host, { name: 'risk-evolution', base: 11.5 }); }
  /** data: {points, nowTtca, futures, tier, key} */
  set(d) {
    const f = d.futures && d.futures.ttca_grid_d ? { ttca_grid_d: d.futures.ttca_grid_d, quantiles: d.futures.quantiles, miss_quantiles: d.futures.miss_quantiles, n_paths: d.futures.n_paths, start_ttca_d: d.futures.start_ttca_d } : null;
    const now = d.nowTtca != null ? Math.round(d.nowTtca / 0.02) * 0.02 : null; // redraw at most every ~0.5 SIM-h
    this.card.draw('riskEvolution', [{ key: d.key, points: (d.points || []).map((p) => ({ ttca: r3(p.ttca), log10: r3(p.log10), reported: r3(p.reported), miss_m: r3(p.miss_m), status: p.status })), nowTtca: now, futures: f, tier: d.tier }]);
  }
}

export class GgRtn {
  constructor(host) { this.card = new PlotCard(host, { name: 'rtn-relative-position', base: 11.5 }); this.sig = new Map(); this.loading = null; }
  /** e: store event (cdms with rtn_m); sigmas fetched once per released-CDM count from /api/conjunctions/{key}/rtn */
  set(e) {
    if (!e) return;
    const cdms = e.cdms.filter(Boolean);
    const tag = `${e.key}:${cdms.length}`;
    if (!this.sig.has(tag) && this.loading !== tag) {
      this.loading = tag;
      api('GET', `/conjunctions/${encodeURIComponent(e.key)}/rtn`).then((r) => {
        const m = new Map((r?.cdms || []).map((c) => [c.cdm_index, c]));
        this.sig.set(tag, m); this.loading = null; this.set(e);
      }).catch(() => { this.sig.set(tag, new Map()); this.loading = null; this.set(e); });
    }
    const sm = this.sig.get(tag) || new Map();
    const rows = cdms.map((c) => {
      const x = sm.get(c.cdm_index);
      const rtn = x?.rtn_m || c.rtn_m;
      return { ttca_d: r3(c.ttca_d ?? x?.ttca_d), rtn_m: rtn ? rtn.map(r3) : null, sigma_comb_m: x?.sigma_comb_m ? x.sigma_comb_m.map(r3) : null };
    }).filter((c) => c.rtn_m && c.ttca_d != null);
    this.card.draw('rtnPlot', [{ key: e.key, cdms: rows }]);
  }
}
