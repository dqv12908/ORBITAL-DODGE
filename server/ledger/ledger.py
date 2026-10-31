"""Append-only, hash-chained audit ledger in SQLite (WAL). Design §6.4.

content_hash = sha256(canonical{kind, event_key, actor, sim_time, payload, refs})
hash         = sha256(prev_hash ‖ seq ‖ ts_wall ‖ content_hash)
genesis prev = sha256("closecall:genesis:" + config_hash + model_sha)
BEFORE UPDATE / DELETE triggers abort, so rows can only be appended.
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.util import canonical, iso, sha256, wall_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
  seq INTEGER PRIMARY KEY,
  ts_wall TEXT NOT NULL,
  sim_time TEXT,
  kind TEXT NOT NULL,
  type TEXT,
  event_key TEXT,
  actor TEXT,
  payload TEXT NOT NULL,
  refs TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_event ON ledger(event_key);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY, scope TEXT NOT NULL, request_hash TEXT NOT NULL, response_json TEXT NOT NULL, status INTEGER NOT NULL, ts_wall TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT PRIMARY KEY, run_id TEXT, event_key TEXT, status TEXT NOT NULL, body TEXT NOT NULL, ts_wall TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policy_versions (
  version TEXT PRIMARY KEY, policy TEXT NOT NULL, actor TEXT, ts_wall TEXT NOT NULL
);
"""


def content_hash(kind: str, event_key: str | None, actor: str | None, sim_time: str | None, payload: Any, refs: Any) -> str:
    return sha256(canonical({"kind": kind, "event_key": event_key, "actor": actor, "sim_time": sim_time,
                             "payload": payload, "refs": refs}))


def chain_hash(prev_hash: str, seq: int, ts_wall: str, chash: str) -> str:
    return sha256(f"{prev_hash}|{seq}|{ts_wall}|{chash}")


class Ledger:
    def __init__(self, path: Path | str, config_hash: str = "", model_sha: str = ""):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        row = self.conn.execute("SELECT v FROM meta WHERE k='genesis'").fetchone()
        if row:
            self.genesis = row["v"]
        else:
            self.genesis = sha256("closecall:genesis:" + config_hash + model_sha)
            self.conn.execute("INSERT INTO meta(k, v) VALUES ('genesis', ?)", (self.genesis,))
        last = self.conn.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        self.seq = int(last["seq"]) if last else 0
        self.head = last["hash"] if last else self.genesis
        self._verified: bool | None = None

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def transaction(self):
        """Group many appends into one SQLite transaction (scenario preload). Rows appended before an error are
        committed, so the in-memory head always matches the table."""
        with self._lock:
            started = not self.conn.in_transaction
            if started:
                self.conn.execute("BEGIN")
        try:
            yield self
        finally:
            if started:
                with self._lock:
                    if self.conn.in_transaction:
                        self.conn.execute("COMMIT")

    def append(self, kind: str, payload: Any, *, event_key: str | None = None, actor: str | None = "system",
               sim_time: str | None = None, refs: dict | None = None, type_: str | None = None) -> dict:
        refs = refs or {}
        with self._lock:
            seq = self.seq + 1
            ts = iso(wall_now())
            chash = content_hash(kind, event_key, actor, sim_time, payload, refs)
            h = chain_hash(self.head, seq, ts, chash)
            self.conn.execute(
                "INSERT INTO ledger(seq, ts_wall, sim_time, kind, type, event_key, actor, payload, refs, content_hash, prev_hash, hash)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (seq, ts, sim_time, kind, type_, event_key, actor, canonical(payload).decode(), canonical(refs).decode(),
                 chash, self.head, h))
            self.seq, self.head = seq, h
            return {"seq": seq, "hash": h, "kind": kind, "content_hash": chash}

    def entries(self, after: int = 0, limit: int = 200, event_key: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 2000))
        q = "SELECT * FROM ledger WHERE seq > ?"
        args: list[Any] = [int(after)]
        if event_key:
            q += " AND event_key = ?"
            args.append(event_key)
        q += " ORDER BY seq LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        import orjson
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = orjson.loads(d["payload"])
            d["refs"] = orjson.loads(d["refs"])
            out.append(d)
        return out

    def content_hashes(self, after: int = 0) -> list[str]:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT content_hash FROM ledger WHERE seq > ? ORDER BY seq", (after,))]

    def verify(self) -> dict:
        """Recompute every content hash and chain link. Returns {ok, n, first_bad_seq, reason}."""
        prev = self.genesis
        n = 0
        expected_seq = 1
        with self._lock:
            rows = self.conn.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
        import orjson
        for r in rows:
            n += 1
            seq = r["seq"]
            if seq != expected_seq:
                return self._bad(n, seq, "sequence gap")
            payload = orjson.loads(r["payload"])
            refs = orjson.loads(r["refs"])
            ch = content_hash(r["kind"], r["event_key"], r["actor"], r["sim_time"], payload, refs)
            if ch != r["content_hash"]:
                return self._bad(n, seq, "content hash mismatch")
            if r["prev_hash"] != prev:
                return self._bad(n, seq, "prev hash mismatch")
            h = chain_hash(prev, seq, r["ts_wall"], ch)
            if h != r["hash"]:
                return self._bad(n, seq, "chain hash mismatch")
            prev = h
            expected_seq += 1
        self._verified = True
        return {"ok": True, "n": n, "first_bad_seq": None, "head": prev}

    def _bad(self, n: int, seq: int, reason: str) -> dict:
        self._verified = False
        return {"ok": False, "n": n, "first_bad_seq": seq, "reason": reason}

    @property
    def verified(self) -> bool | None:
        return self._verified

    def export_jsonl(self):
        from app.util import dumps
        after = 0
        while True:
            batch = self.entries(after=after, limit=1000)
            if not batch:
                break
            for e in batch:
                yield dumps(e) + "\n"
            after = batch[-1]["seq"]

    # ---- side tables (mutable, not part of the chain) -------------------
    def idem_get(self, key: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute("SELECT * FROM idempotency WHERE key=?", (key,)).fetchone()

    def idem_put(self, key: str, scope: str, request_hash: str, response_json: str, status: int) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO idempotency(key, scope, request_hash, response_json, status, ts_wall)"
                              " VALUES (?,?,?,?,?,?)", (key, scope, request_hash, response_json, status, iso(wall_now())))

    def approval_put(self, approval_id: str, run_id: str, event_key: str, status: str, body_json: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO approvals(approval_id, run_id, event_key, status, body, ts_wall)"
                              " VALUES (?,?,?,?,?,?)", (approval_id, run_id, event_key, status, body_json, iso(wall_now())))

    def policy_put(self, version: str, policy_json: str, actor: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR IGNORE INTO policy_versions(version, policy, actor, ts_wall) VALUES (?,?,?,?)",
                              (version, policy_json, actor, iso(wall_now())))

    def close(self) -> None:
        with self._lock:
            self.conn.close()
