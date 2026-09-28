from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

GENESIS_PREV = "0" * 64


def _entry_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class AuditLog:
    """Append-only, hash-chained JSONL record of every decision, including denials.

    Citta holds executed conduct (the identity ledger); the audit log holds everything the
    harness decided, so blocked and denied proposals are accountable too.
    """

    def __init__(self, path: str | Path, clock: Callable[[], str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._seq, self._prev = 0, GENESIS_PREV
        if self.path.exists():
            for entry in self.entries():
                self._seq, self._prev = entry["seq"], entry["hash"]

    def record(self, event: str, agent_did: str, **data: Any) -> dict[str, Any]:
        body = {
            "seq": self._seq + 1, "ts": self.clock(), "event": event,
            "agent_did": agent_did, "data": data, "prev_hash": self._prev,
        }
        entry = {**body, "hash": _entry_hash(body)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        self._seq, self._prev = entry["seq"], entry["hash"]
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def verify(self) -> bool:
        prev, seq = GENESIS_PREV, 0
        for entry in self.entries():
            body = {key: value for key, value in entry.items() if key != "hash"}
            if entry["seq"] != seq + 1 or entry["prev_hash"] != prev or entry["hash"] != _entry_hash(body):
                return False
            prev, seq = entry["hash"], entry["seq"]
        return True
