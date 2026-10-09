import hashlib

from jiva_harness.citta import CittaLedger, Experience


def _pair(left: str, right: str) -> str:
    return hashlib.sha256(bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def _full_rebuild_root(leaves: list[str]) -> str:
    """The original O(n) root: rebuild every level, pairing an odd level's last node with itself."""
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def _experience(i: int) -> Experience:
    return Experience({"i": i}, "goal", {}, {"name": "act", "i": i}, {"total": 1.0}, {"digest": str(i)})


def test_incremental_root_and_proofs_match_full_rebuild(tmp_path):
    ledger = CittaLedger.create(tmp_path / "citta.jsonl", [{"stage": "brahmacarya"}])
    assert ledger.root == _full_rebuild_root([ledger.genesis_hash])
    for i in range(300):
        ledger.append(_experience(i))
        leaves = [ledger.genesis_hash] + [r.leaf_hash for r in ledger.records]
        assert ledger.root == _full_rebuild_root(leaves)
    for index in (0, 1, 2, 127, 128, 255, 299, 300):
        proof = ledger.prove(index)
        assert proof.verify() and proof.expected_root == ledger.root


def test_reopened_ledger_has_same_root(tmp_path):
    buffer = [{"stage": "brahmacarya"}]
    ledger = CittaLedger.create(tmp_path / "citta.jsonl", buffer)
    for i in range(37):
        ledger.append(_experience(i))
    assert CittaLedger.open(tmp_path / "citta.jsonl", buffer).root == ledger.root
