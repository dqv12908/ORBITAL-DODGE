"""CSV (train + test) -> data/processed/cdm_clean.parquet (sentinel -> NaN + cov_unknown flags, event_key, cdm_index)."""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import time

import pandas as pd

from riskengine import cdm, paths


def main():
    t0 = time.time()
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    parts = [cdm.clean(cdm.load_raw(s), s) for s in ("train", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["c_object_type"] = df["c_object_type"].astype("string")
    df.to_parquet(paths.CDM_CLEAN, index=False)
    n_cov = int((df.cov_unknown_t | df.cov_unknown_c).sum())
    print(f"[ingest] {len(df)} rows, {df.event_key.nunique()} events, cov_unknown rows {n_cov} "
          f"-> {paths.CDM_CLEAN} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
