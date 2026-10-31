# Live sky: screening today's public catalogue

Run: 72 h window from 2026-10-31T04:00Z, 10 s grid, on triologay. Output: `data/livesky/latest.json`.

## Method
- **Objects:** 28,686 of 30,654 catalogued objects with perigee < 2000 km and element sets < 30 days old. Sources: CelesTrak GP `active`, the three debris groups, name searches (DEB, R/B, COSMOS, OBJECT, OPS, METEOR), SATCAT, SOCRATES `sort-minRange.csv`. One download each. These cover 94 % of SOCRATES pairs.
- **Propagation:** SGP4 from OMM (python-sgp4 `SatrecArray`), so objects without TLEs are included. About 7.4e8 propagations across 52 processes, using float64 shared memory. Takes 21.5 s.
- **Screening:** sorted-grid spatial hash on the GPU (86 km cells, PG503 32 GB, peak 3.4 GB) at each time step. Takes 21.6 s.
- **Closest approach:** each candidate's time and minimum range are refined with a linear relative-motion formula, giving about 2 m error at 10 s steps. The closest-approach point is converted to latitude, longitude and altitude.
- **Max Pc:** the worst case over an isotropic Gaussian of unknown size, with hard-body radius from the SATCAT radar cross-section. Labelled "max Pc (no covariance in public TLEs)".
- **Total runtime:** 156 s, including max Pc and validation.

## Results
- 334,820 events < 5 km across 193,319 pairs in 72 h. 24,710 of them are < 1 km.
- **Next 24 h:** 4,402 events < 1 km (2,496 excluding formation pairs) and 75,820 < 5 km.
- **Formation pairs:** 398 formation or co-located pairs (relative speed < 0.1 km/s, e.g. TanDEM-X/TerraSAR-X) are counted in `summary` but kept off the top list.
- **Top list:** `latest.json` holds the closest encounter for each of 200 pairs with relative speed ≥ 0.1 km/s. 184 are payload–payload, mostly Starlink–Starlink.

## Validation against SOCRATES
Matching rule: same NORAD pair, TCA within ±60 s. 80,839 SOCRATES events fall in our window.

| Measure | Result |
|---|---|
| Recall, all events < 5 km / < 1 km | 28.8 % / 33.9 % |
| Recall where we used the same element sets as SOCRATES (epoch = TCA − DSE, n = 1,746) | **97.1 %** (< 5 km), **88.0 %** (< 1 km) |
| Median range difference, same element sets | **0.3 m** |
| Median ΔTCA / Δrange, all matches | 0.10 s / 0.19 km |
| Precision, payload events | 7.5 % overall, 32.9 % on same element sets |
| Precision, same element sets, excluding Starlink–Starlink and formation pairs | 91.0 % |
| Top-200 list matched to SOCRATES | 6 % |

**What this means:**
- With the same inputs, our screening reproduces SOCRATES.
- Most disagreement comes from element age. SOCRATES used elements about 1.8 days old; ours are newer. For the same pairs, our SGP4 propagation gives 8–23 km where SOCRATES gives 0.1–0.5 km.
- Starlink–Starlink pairs are 79 % of our payload events but only 32 % of SOCRATES's. SOCRATES appears to leave many of them out; we could not confirm why.

## Limits
- Public elements carry no covariance, so Pc is only a worst-case bound. SGP4 errors are about a kilometre or more, so this checks the screening, not the truth.
- About 1–1.2k SOCRATES objects, mostly inactive payloads, are not in our snapshot.
- Pairs drifting at about 1 m/s have a poorly defined closest-approach time. We report one closest point where SOCRATES reports several.
- The elements date from 30–31 Oct 2026. To refresh, download `cache/gp/` once after the next CelesTrak update and rerun.
- This layer is context for the globe and the 24 h dashboard. It is never used to compute risk for the dataset events.

## Rerun (on triologay)
```bash
cd /root/orbital-dodge/research/livesky
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 .venv-sky/bin/python screen.py --hours 72
```
Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, CUDA numbers the cards differently from nvidia-smi: index 3 is then the RTX 2060.
Env: `.venv-sky` (Python 3.12, torch 2.6.0+cu124, sgp4 2.27).
