from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from .audit import AuditLog
from .citta import Experience
from .principles import ActionContext, PrincipleHook
from .runtime import JivaHarness, PrincipalRequest
from .sealing import PlaintextSealer, Sealer
from .tools import PermissionDenied, ToolRegistry


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Finish:
    answer: str


Proposal = ToolCall | Finish
Outcome = Literal["executed", "blocked", "denied", "error", "halted"]


@dataclass(frozen=True)
class Turn:
    step: int
    call: ToolCall
    outcome: Outcome
    observation: dict[str, Any]


class Policy(Protocol):
    """Sankalpa's proposal boundary. An LLM adapter implements this; the harness never trusts it."""

    def propose(self, goal: str, history: list[Turn], tools: list[dict[str, Any]]) -> Proposal: ...


class ScriptedPolicy:
    """Deterministic stand-in for a model: replays a fixed list of proposals."""

    def __init__(self, proposals: list[Proposal]):
        self._proposals = list(proposals)

    def propose(self, goal: str, history: list[Turn], tools: list[dict[str, Any]]) -> Proposal:
        return self._proposals[len(history)] if len(history) < len(self._proposals) else Finish("script exhausted")


@dataclass(frozen=True)
class LoopResult:
    answer: str | None
    stop_reason: Literal["finished", "max_steps", "halted"]
    turns: list[Turn] = field(default_factory=list)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AgentLoop:
    """propose -> principle hook -> permission gate -> execute -> critic -> Citta + audit, repeated.

    Refused actions (blocked, denied, halted) are Citta conduct too. Their tool, args and reason go
    through `sealer`, so they can be encrypted later without changing the ledger.
    """

    def __init__(
        self, harness: JivaHarness, registry: ToolRegistry, hook: PrincipleHook,
        audit: AuditLog, policy: Policy, max_steps: int = 8, sealer: Sealer | None = None,
    ):
        self.harness, self.registry, self.hook = harness, registry, hook
        self.audit, self.policy, self.max_steps = audit, policy, max_steps
        self.sealer: Sealer = sealer or PlaintextSealer()
        self._halt: tuple[str, str] | None = None
        # Observers for interactive front ends: called before a proposal is gated and after its verdict.
        self.on_proposal: Callable[[int, ToolCall], None] | None = None
        self.on_turn: Callable[[Turn], None] | None = None

    def halt(self, raised_by: str, reason: str) -> None:
        """Algedonic loop: any principal or subsystem may stop the run; checked around every step."""
        self._halt = (raised_by, reason)

    def run(
        self, principal: str, channel: str, goal: str,
        approved_irreversible: frozenset[str] = frozenset(),
    ) -> LoopResult:
        did = self.harness.citta.did
        self.audit.record("run_started", did, principal=principal, channel=channel, goal=goal,
                          approved_irreversible=sorted(approved_irreversible))
        turns: list[Turn] = []
        self._halt = None
        for step in range(1, self.max_steps + 1):
            proposal = self.policy.propose(goal, turns, self.registry.specs())
            if self._halt:  # a halt raised while the model was thinking beats its proposal
                return self._stop_halted(step, turns)
            if isinstance(proposal, Finish):
                self.audit.record("run_finished", did, step=step, answer=proposal.answer,
                                  citta_root=self.harness.citta.root)
                return LoopResult(proposal.answer, "finished", turns)
            self.audit.record("proposal", did, step=step, tool=proposal.tool, args=proposal.args)
            if self.on_proposal:
                self.on_proposal(step, proposal)
            turn = self._step(step, proposal, principal, channel, goal, approved_irreversible)
            turns.append(turn)
            if self.on_turn:
                self.on_turn(turn)
            if self._halt:
                return self._stop_halted(step, turns)
        self.audit.record("run_stopped", did, reason="max_steps", citta_root=self.harness.citta.root)
        return LoopResult(None, "max_steps", turns)

    def _stop_halted(self, step: int, turns: list[Turn]) -> LoopResult:
        raised_by, reason = self._halt or ("", "")
        self.audit.record("algedonic_halt", self.harness.citta.did, step=step, raised_by=raised_by,
                          reason=reason, citta_root=self.harness.citta.root)
        return LoopResult(None, "halted", turns)

    def _constraints(self) -> dict[str, Any]:
        spec = self.harness.spec
        return {
            "allowed_actions": list(spec.capabilities), "conditions": spec.conditions,
            "guna": spec.guna, "universal_obligations": list(spec.universal_obligations),
            "purpose": spec.purpose, "principles": [p.name for p in self.hook.principles],
        }

    def _step(
        self, step: int, call: ToolCall, principal: str, channel: str, goal: str,
        approved_irreversible: frozenset[str],
    ) -> Turn:
        turn = self._decide(step, call, principal, channel, goal, approved_irreversible)
        if turn.outcome in ("blocked", "denied", "halted"):
            self._record_refusal(turn, principal, channel, goal)
        return turn

    def _record_refusal(self, turn: Turn, principal: str, channel: str, goal: str) -> None:
        """Citta leaf for an action that did not run. Plaintext fields say only that a refusal happened."""
        state = {"principal": principal, "channel": channel, "step": turn.step}
        constraints = self._constraints()
        action = {
            "name": "refused", "outcome": turn.outcome,
            "sealed": self.sealer.seal({"tool": turn.call.tool, "args": turn.call.args,
                                        "reason": turn.observation.get("error", "")}),
        }
        reward = {"evaluated": False, "reason": "action was not executed"}
        trace = {"digest": _digest({"state": state, "goal": goal, "constraints": constraints, "action": action})}
        record = self.harness.citta.append(Experience(state, goal, constraints, action, reward, trace))
        self.audit.record("refusal_recorded", self.harness.citta.did, step=turn.step, outcome=turn.outcome,
                          seal_scheme=action["sealed"]["scheme"], citta_index=record.index,
                          citta_leaf=record.leaf_hash)

    def _decide(
        self, step: int, call: ToolCall, principal: str, channel: str, goal: str,
        approved_irreversible: frozenset[str],
    ) -> Turn:
        harness, did = self.harness, self.harness.citta.did
        try:
            tool = self.registry.get(call.tool)
        except PermissionDenied as denied:
            self.audit.record("permission_denied", did, step=step, reason=str(denied))
            return Turn(step, call, "denied", {"error": str(denied)})

        # 1. Sadharana-dharma: universal principles, before anything runs.
        verdicts = self.hook.evaluate(ActionContext(goal, principal, tool, call.args, step, approved_irreversible))
        self.audit.record("principle_check", did, step=step,
                          verdicts=[{"principle": v.principle, "allowed": v.allowed, "reason": v.reason, "halt": v.halt}
                                    for v in verdicts])
        halting = [v for v in verdicts if v.halt]
        if halting:
            self.halt(halting[0].principle, halting[0].reason)
            return Turn(step, call, "halted", {"error": f"{halting[0].principle}: {halting[0].reason}"})
        failed = [v for v in verdicts if not v.allowed]
        if failed:
            return Turn(step, call, "blocked", {"error": "; ".join(f"{v.principle}: {v.reason}" for v in failed)})

        # 2. Visesha-dharma: scoped, ephemeral, identity-bound credential for this call only.
        try:
            credential = self.registry.gate(tool, call.args, harness.spec, principal, channel, goal, did)
        except PermissionDenied as denied:
            self.audit.record("permission_denied", did, step=step, tool=tool.name, reason=str(denied))
            return Turn(step, call, "denied", {"error": str(denied)})
        self.audit.record("credential_issued", did, step=step, tool=tool.name, capability=credential.capability,
                          ephemeral=credential.ephemeral, transferable=credential.transferable)

        # 3. Prakriti executes.
        try:
            observation = tool.handler(dict(call.args))
        except Exception as exc:  # tool failure is an observation, not a harness crash
            self.audit.record("tool_error", did, step=step, tool=tool.name, error=repr(exc))
            return Turn(step, call, "error", {"error": repr(exc)})

        # 4. Karmaphaladata evaluates; Citta records the paper's tuple.
        constraints = self._constraints()
        action = {"name": credential.capability, "tool": tool.name, "args": call.args,
                  "content": f"{harness.spec.purpose} via {tool.name}"}
        state = {"partial_observation": call.args, "principal": principal, "channel": channel, "step": step}
        trace = {"digest": _digest({"state": state, "goal": goal, "constraints": constraints, "action": action})}
        request = PrincipalRequest(principal, channel, goal, call.args, tool.capability)
        reward = harness.karmaphaladata.evaluate(harness.spec, request, action, trace)
        record = harness.citta.append(Experience(state, goal, constraints, action, reward.to_dict(), trace))
        self.audit.record("tool_executed", did, step=step, tool=tool.name, observation=observation,
                          reward=reward.to_dict(), citta_index=record.index, citta_leaf=record.leaf_hash)
        return Turn(step, call, "executed", observation)
