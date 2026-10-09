"""Citta importance gate: decide which executed actions are significant enough to become ledger leaves.

Without it, every action is a Citta leaf and a long-running agent's ledger grows without bound. Skipped
actions are not lost: they stay in the hash-chained audit log, and the loop commits to each run's skipped
span with one rollup leaf. Refusals never reach this gate; they are always recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .tools import Tool


@dataclass(frozen=True)
class ImportanceDecision:
    keep: bool
    score: float
    reason: str


class ImportanceModel(Protocol):
    """A cheap decision model (Jev-style, or a small local classifier): P(this action matters)."""

    def probability(self, features: dict[str, Any]) -> float: ...


def features(tool: Tool, step: int, reward: dict[str, Any]) -> dict[str, Any]:
    """What a model sees. Only plaintext metadata, never args or observations, so nothing sealed leaks."""
    return {
        "tool": tool.name, "risk": tool.risk, "capability": tool.capability, "step": step,
        "reward_total": reward.get("total"), "harm": bool(reward.get("harm")),
    }


class ImportanceGate:
    """Rules first, model second. The rules are a floor: a model can add leaves, never drop these.

    Always kept: any non-read action (it changed something), anything a harm detector flagged, and any
    action whose reward fell below `min_reward`. Read-only actions with a nominal reward are skipped unless
    `model` scores them at or above `threshold`.
    """

    def __init__(self, model: ImportanceModel | None = None, threshold: float = 0.5, min_reward: float = 1.0):
        self.model, self.threshold, self.min_reward = model, threshold, min_reward

    def decide(self, tool: Tool, step: int, reward: dict[str, Any]) -> ImportanceDecision:
        if tool.risk != "read":
            return ImportanceDecision(True, 1.0, f"{tool.risk} action")
        if reward.get("harm"):
            return ImportanceDecision(True, 1.0, "flagged by harm detector")
        total = reward.get("total")
        if total is not None and total < self.min_reward:
            return ImportanceDecision(True, 1.0, f"reward {total:.3f} below {self.min_reward}")
        if self.model is None:
            return ImportanceDecision(False, 0.0, "routine read")
        score = float(self.model.probability(features(tool, step, reward)))
        if score >= self.threshold:
            return ImportanceDecision(True, score, f"model score {score:.3f} >= {self.threshold}")
        return ImportanceDecision(False, score, f"model score {score:.3f} < {self.threshold}")
