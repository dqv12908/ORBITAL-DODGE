"""Approval state machine: pending -> approved | rejected | expired. Human decides; idempotent at the API layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Approval:
    approval_id: str
    decision_id: str
    event_key: str
    cdm_index: int
    options: list[str]
    suggested: str | None
    verdict: str
    reasons: list[str]
    ttca_h: float | None
    mcp_passed: bool
    created_sim: str
    status: str = "pending"
    decided: dict | None = field(default=None)
    expired_reason: str | None = None
    primary_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ApprovalError(Exception):
    def __init__(self, code: str, message: str, status: int, extra: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra or {}


class Approvals:
    def __init__(self):
        self.items: dict[str, Approval] = {}

    def reset(self) -> None:
        self.items = {}

    def add(self, a: Approval) -> Approval:
        self.items[a.approval_id] = a
        return a

    def pending(self) -> list[Approval]:
        return [a for a in self.items.values() if a.status == "pending"]

    def for_event(self, key: str) -> list[Approval]:
        return [a for a in self.items.values() if a.event_key == key]

    def expire_event(self, key: str, reason: str) -> list[Approval]:
        out = []
        for a in self.items.values():
            if a.event_key == key and a.status == "pending":
                a.status = "expired"
                a.expired_reason = reason
                out.append(a)
        return out

    def decide(self, approval_id: str, action: str, actor: str, note: str | None, sim_time: str, wait_sim_min: float) -> Approval:
        a = self.items.get(approval_id)
        if a is None:
            raise ApprovalError("NOT_FOUND", f"no approval {approval_id}", 404)
        if a.status != "pending":
            raise ApprovalError("APPROVAL_ALREADY_DECIDED", f"approval is {a.status}", 409,
                                {"approval": a.to_dict()})
        valid = list(a.options) + ["REJECT"]
        if action not in valid:
            raise ApprovalError("INVALID_ACTION", f"action must be one of {valid}", 400, {"valid": valid})
        a.status = "rejected" if action == "REJECT" else "approved"
        a.decided = {"action": action, "actor": actor, "note": note, "sim_time": sim_time, "wait_sim_min": wait_sim_min}
        return a
