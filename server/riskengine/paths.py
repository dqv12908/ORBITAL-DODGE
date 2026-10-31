"""Repository paths (override the repo root with C1_REPO_ROOT, raw data dir with C1_RAW_DIR)."""
import os
from pathlib import Path

REPO = Path(os.environ.get("C1_REPO_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = Path(os.environ.get("C1_RAW_DIR", REPO / "data" / "raw" / "Challenge 1 Airbus "))
PROCESSED = REPO / "data" / "processed"
REGISTRY = REPO / "registry"
FORECASTER_DIR = REGISTRY / "tier_forecaster"
PREDICTIONS = REPO / "predictions"
SCENARIOS = REPO / "shared" / "scenarios"

CDM_CLEAN = PROCESSED / "cdm_clean.parquet"
PC_ROWS = PROCESSED / "pc_rows.parquet"
EVENTS = PROCESSED / "events.parquet"
GEOMETRY = PROCESSED / "geometry.parquet"


def current_forecaster_dir() -> Path:
    cur = FORECASTER_DIR / "CURRENT"
    ver = cur.read_text().strip() if cur.exists() else "1.0.0"
    return FORECASTER_DIR / ver
