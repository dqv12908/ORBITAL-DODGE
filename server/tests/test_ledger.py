"""Ledger: tamper detection reports the bad seq; UPDATE/DELETE are refused by triggers."""
import sqlite3

import pytest

from ledger.ledger import Ledger


def make(tmp_path, n=5):
    led = Ledger(tmp_path / "l.sqlite", config_hash="c", model_sha="m")
    for i in range(n):
        led.append("signal", {"i": i, "x": [1, 2, 3]}, event_key="train:1", actor="engine", sim_time=f"2026-10-31T0{i}:00:00Z")
    return led


def test_chain_verifies(tmp_path):
    led = make(tmp_path)
    assert led.verify() == {"ok": True, "n": 5, "first_bad_seq": None, "head": led.head}


def test_update_and_delete_raise(tmp_path):
    led = make(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        led.conn.execute("UPDATE ledger SET payload='{}' WHERE seq=2")
    with pytest.raises(sqlite3.DatabaseError):
        led.conn.execute("DELETE FROM ledger WHERE seq=2")


def test_tamper_detected_at_seq(tmp_path):
    led = make(tmp_path)
    # an attacker with file access drops the trigger and edits row 3
    led.conn.execute("DROP TRIGGER ledger_no_update")
    led.conn.execute("UPDATE ledger SET payload='{\"i\":99,\"x\":[1,2,3]}' WHERE seq=3")
    r = led.verify()
    assert r["ok"] is False and r["first_bad_seq"] == 3


def test_tamper_rehash_detected_by_chain(tmp_path):
    led = make(tmp_path)
    led.conn.execute("DROP TRIGGER ledger_no_update")
    # recompute content hash for the edited payload but leave the chain: chain hash must catch it
    from ledger.ledger import content_hash
    ch = content_hash("signal", "train:1", "engine", "2026-10-31T01:00:00Z", {"i": 42, "x": [1, 2, 3]}, {})
    led.conn.execute("UPDATE ledger SET payload='{\"i\":42,\"x\":[1,2,3]}', content_hash=? WHERE seq=2", (ch,))
    r = led.verify()
    assert r["ok"] is False and r["first_bad_seq"] == 2


def test_reopen_continues_chain(tmp_path):
    led = make(tmp_path, 3)
    led.close()
    led2 = Ledger(tmp_path / "l.sqlite", config_hash="other", model_sha="other")
    led2.append("human", {"a": 1})
    assert led2.verify()["ok"] and led2.seq == 4
