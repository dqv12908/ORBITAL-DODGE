"""Model registry access: list models, read cards and metrics, verify artifact sha256 against the model card."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_shas(card: dict) -> dict[str, str]:
    """Card may list shas as {artifacts: {file: sha}} or {artifact_sha256: {file: sha}} or {artifact_sha256: sha}."""
    for k in ("artifacts", "artifact_sha256", "sha256", "files"):
        v = card.get(k)
        if isinstance(v, dict):
            return {f: (s.get("sha256") if isinstance(s, dict) else s) for f, s in v.items() if s}
        if isinstance(v, str) and len(v) == 64:
            return {"model.joblib": v}
    return {}


def verify_model_dir(path: Path) -> bool | None:
    """True if every sha listed in model_card.json matches, False on mismatch, None if nothing to check."""
    card_p = Path(path) / "model_card.json"
    if not card_p.exists():
        return None
    try:
        card = json.loads(card_p.read_text())
    except Exception:
        return False
    exp = _expected_shas(card)
    if not exp:
        return None
    for fname, sha in exp.items():
        fp = Path(path) / fname
        if not fp.exists() or _sha(fp) != sha:
            return False
    return True


def list_models(repo_dir: Path) -> list[dict]:
    reg = Path(repo_dir) / "registry"
    out = []
    if not reg.exists():
        return out
    for d in sorted(p for p in reg.iterdir() if p.is_dir()):
        cur = d / "CURRENT"
        versions = sorted(p.name for p in d.iterdir() if p.is_dir())
        current = cur.read_text().strip() if cur.exists() else (versions[-1] if versions else None)
        card = read_card(repo_dir, d.name) or {}
        out.append({"name": d.name, "current": current, "versions": versions,
                    "has_card": bool(card), "sha_ok": verify_model_dir(d / current) if current else None,
                    "created_at": card.get("created_at"), "intended_use": card.get("intended_use")})
    return out


def read_card(repo_dir: Path, name: str) -> dict | None:
    d = Path(repo_dir) / "registry" / name
    if not d.exists() or "/" in name or ".." in name:
        return None
    cur = d / "CURRENT"
    cands = []
    if cur.exists():
        cands.append(d / cur.read_text().strip() / "model_card.json")
    cands.append(d / "model_card.json")
    for c in cands:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                return {"error": f"unreadable {c.name}"}
    return None


def collect_eval(repo_dir: Path) -> dict:
    """Merge registry/*/metrics.json (and */CURRENT version dirs) plus research/bigml/results.json."""
    repo_dir = Path(repo_dir)
    out: dict = {"sources": [], "metrics": {}, "reliability": {}, "missing": []}
    reg = repo_dir / "registry"
    if reg.exists():
        for d in sorted(p for p in reg.iterdir() if p.is_dir()):
            files = []
            cur = d / "CURRENT"
            if cur.exists():
                files.append(d / cur.read_text().strip())
            files.append(d)
            for base in files:
                m = base / "metrics.json"
                if m.exists() and d.name not in out["metrics"]:
                    try:
                        out["metrics"][d.name] = json.loads(m.read_text())
                        out["sources"].append(str(m.relative_to(repo_dir)))
                    except Exception as e:
                        out["missing"].append(f"{m.relative_to(repo_dir)}: {e}")
                r = base / "reliability.json"
                if r.exists() and d.name not in out["reliability"]:
                    try:
                        out["reliability"][d.name] = json.loads(r.read_text())
                    except Exception:
                        pass
    bm = repo_dir / "research" / "bigml" / "results.json"
    if bm.exists():
        try:
            out["metrics"]["bigml"] = json.loads(bm.read_text())
            out["sources"].append("research/bigml/results.json")
        except Exception as e:
            out["missing"].append(f"research/bigml/results.json: {e}")
    if not out["metrics"]:
        out["missing"].append("no registry metrics yet (engine pipelines not run)")
    return out
