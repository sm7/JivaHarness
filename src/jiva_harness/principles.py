from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .tools import CONSEQUENTIAL, Tool


@dataclass(frozen=True)
class ActionContext:
    """What a principle may inspect before an action runs."""

    goal: str
    principal: str
    tool: Tool
    args: dict[str, Any]
    step: int
    approved_irreversible: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Verdict:
    principle: str
    allowed: bool
    reason: str
    halt: bool = False  # algedonic signal: stop the whole run, not just this action


class Principle(Protocol):
    name: str

    def check(self, ctx: ActionContext) -> Verdict: ...


class Ahimsa:
    """Non-harm: no irreversible or external-effect action unless the human principal approved that tool.

    Approval comes from the run's principal, never from tool args, because args are model-authored:
    either up front (`approved_irreversible`) or, when `ask` is given, by asking the principal at the
    moment the action is proposed.
    """

    name = "ahimsa"

    def __init__(self, ask: Callable[[ActionContext], bool] | None = None):
        self.ask = ask

    def check(self, ctx: ActionContext) -> Verdict:
        if ctx.tool.risk not in CONSEQUENTIAL:
            return Verdict(self.name, True, f"{ctx.tool.risk} action")
        if ctx.tool.name in ctx.approved_irreversible:
            return Verdict(self.name, True, f"{ctx.tool.risk} action approved by principal")
        if self.ask is not None and self.ask(ctx):
            return Verdict(self.name, True, f"{ctx.tool.risk} action approved by principal when proposed")
        return Verdict(self.name, False, f"'{ctx.tool.name}' is {ctx.tool.risk} and was not approved by the principal")


class PrincipleHook:
    """Sadharana-dharma gate run before every action. Every principle is evaluated and recorded."""

    def __init__(self, principles: list[Principle]):
        self.principles = list(principles)

    def evaluate(self, ctx: ActionContext) -> list[Verdict]:
        return [principle.check(ctx) for principle in self.principles]
