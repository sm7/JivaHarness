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
        self.records = records or []

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
        return self._root([self.genesis_hash] + [record.leaf_hash for record in self.records])

    @staticmethod
    def _root(leaves: list[str]) -> str:
        level = list(leaves)
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        return level[0]

    def append(self, experience: Experience) -> CittaRecord:
        # The paper's Lt = H(st, gt, ct, at, rt, zt), represented canonically by fields.
        record = CittaRecord(len(self.records) + 1, _hash(experience.to_dict()), experience)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "type": "experience", "index": record.index,
                "leaf_hash": record.leaf_hash, "experience": experience.to_dict(),
            }, sort_keys=True) + "\n")
        self.records.append(record)
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
        leaves = [self.genesis_hash] + [record.leaf_hash for record in self.records]
        if not 0 <= record_index < len(leaves):
            raise IndexError(record_index)
        index = record_index
        level = leaves
        siblings: list[tuple[str, str]] = []
        while len(level) > 1:
            if len(level) % 2:
                level = level + [level[-1]]
            sibling_index = index - 1 if index % 2 else index + 1
            siblings.append(("left" if index % 2 else "right", level[sibling_index]))
            index //= 2
            level = [_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        return MerkleProof(leaves[record_index], record_index, tuple(siblings), level[0])
