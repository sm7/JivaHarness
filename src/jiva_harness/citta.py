from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pair(left: str, right: str) -> str:
    return hashlib.sha256(bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


@dataclass(frozen=True)
class Experience:
    state: dict[str, Any]
    goal: str
    constraints: dict[str, Any]
    action: dict[str, Any]
    reward: dict[str, Any]
    activation_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CittaRecord:
    index: int
    leaf_hash: str
    experience: Experience


@dataclass(frozen=True)
class MerkleProof:
    leaf_hash: str
    index: int
    siblings: tuple[tuple[str, str], ...]
    expected_root: str

    def verify(self) -> bool:
        current = self.leaf_hash
        for side, sibling in self.siblings:
            current = _pair(sibling, current) if side == "left" else _pair(current, sibling)
        return current == self.expected_root


class CittaLedger:
    """Persistent conduct ledger; stable identity is its brahmacarya genesis hash."""

    def __init__(self, path: Path, genesis_hash: str, records: list[CittaRecord] | None = None):
        self.path = path
        self.genesis_hash = genesis_hash
        self.records: list[CittaRecord] = []
        # Merkle levels, leaves first. Appends touch one node per level, so root and proofs are O(log n).
        self._levels: list[list[str]] = [[genesis_hash]]
        for record in records or []:
            self._add_record(record)

    @classmethod
    def create(cls, path: str | Path, brahmacarya_buffer: list[dict[str, Any]]) -> "CittaLedger":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        genesis_hash = _hash(brahmacarya_buffer)
        path.write_text(json.dumps({"type": "genesis", "genesis_hash": genesis_hash}, sort_keys=True) + "\n", encoding="utf-8")
        return cls(path, genesis_hash)

    @classmethod
    def open(cls, path: str | Path, brahmacarya_buffer: list[dict[str, Any]]) -> "CittaLedger":
        """Resume an existing ledger (same identity, same conduct) or create it if absent."""
        path = Path(path)
        if not path.exists():
            return cls.create(path, brahmacarya_buffer)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        genesis_hash = _hash(brahmacarya_buffer)
        if not lines or lines[0] != {"type": "genesis", "genesis_hash": genesis_hash}:
            raise ValueError(f"{path} belongs to a different identity (genesis mismatch)")
        records = [CittaRecord(row["index"], row["leaf_hash"], Experience(**row["experience"])) for row in lines[1:]]
        ledger = cls(path, genesis_hash, records)
        if not ledger.verify():
            raise ValueError(f"{path} failed verification; refusing to extend a tampered ledger")
        return ledger

    @property
    def did(self) -> str:
        return f"did:jiva:{self.genesis_hash}"

    @property
    def root(self) -> str:
        return self._levels[-1][0]

    def _add_record(self, record: CittaRecord) -> None:
        """Add a leaf and recompute only its path to the root. An odd level pairs its last node with itself."""
        self.records.append(record)
        self._levels[0].append(record.leaf_hash)
        level = 0
        while len(self._levels[level]) > 1:
            nodes = self._levels[level]
            parent = (len(nodes) - 1) // 2
            left = nodes[2 * parent]
            right = nodes[2 * parent + 1] if 2 * parent + 1 < len(nodes) else left
            if level + 1 == len(self._levels):
                self._levels.append([])
            above = self._levels[level + 1]
            if parent < len(above):
                above[parent] = _pair(left, right)
            else:
                above.append(_pair(left, right))
            level += 1

    def append(self, experience: Experience) -> CittaRecord:
        # The paper's Lt = H(st, gt, ct, at, rt, zt), represented canonically by fields.
        record = CittaRecord(len(self.records) + 1, _hash(experience.to_dict()), experience)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "type": "experience", "index": record.index,
                "leaf_hash": record.leaf_hash, "experience": experience.to_dict(),
            }, sort_keys=True) + "\n")
        self._add_record(record)
        return record

    def verify(self) -> bool:
        if not self.path.exists():
            return False
        lines = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        if not lines or lines[0] != {"type": "genesis", "genesis_hash": self.genesis_hash}:
            return False
        if len(lines) != len(self.records) + 1:
            return False
        return all(
            row["index"] == record.index
            and row["leaf_hash"] == _hash(row["experience"])
            and row["leaf_hash"] == record.leaf_hash
            for row, record in zip(lines[1:], self.records)
        )

    def prove(self, record_index: int) -> MerkleProof:
        if not 0 <= record_index < len(self._levels[0]):
            raise IndexError(record_index)
        index = record_index
        siblings: list[tuple[str, str]] = []
        for nodes in self._levels[:-1]:
            sibling_index = index - 1 if index % 2 else index + 1
            sibling = nodes[sibling_index] if sibling_index < len(nodes) else nodes[index]
            siblings.append(("left" if index % 2 else "right", sibling))
            index //= 2
        return MerkleProof(self._levels[0][record_index], record_index, tuple(siblings), self.root)
