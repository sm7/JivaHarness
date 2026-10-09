"""Interactive harness session: the principal types tasks, the model proposes, the harness decides live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .demo import SCENARIOS, TOOLS, build_loop, decided_by, ledger_status
from .loop import ToolCall, Turn
from .principles import ActionContext

HELP = """Type a task for the agent. Commands:
  /tools            what the agent can call and how each is governed
  /approve TOOL     pre-approve an irreversible tool for the rest of the session
  /revoke TOOL      withdraw that approval; you will be asked again
  /new              start a new conversation (same identity and ledger)
  /scenarios        ready-made tasks; /run NAME runs one
  /ledger           verify Citta and the audit log
  /quit             exit (Ctrl-D works too)"""


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={json.dumps(value)}" for key, value in sorted(args.items()))


class Session:
    """One Jiva identity and one conversation across many tasks, with the principal in the loop.

    Irreversible actions the principal has not approved are put to them as they are proposed (Ahimsa
    asks); a "no" is recorded as a block. The ledger in state_dir persists between sessions.
    """

    def __init__(
        self, policy: Any, state_dir: Path, read: Callable[[str], str] = input,
        write: Callable[[str], None] = print, recoverable: Callable[[Exception], str | None] = lambda exc: None,
        max_steps: int = 8,
    ):
        self.policy, self.state_dir, self.read, self.write = policy, state_dir, read, write
        self.recoverable = recoverable
        self.always: set[str] = set()
        self.loop, self.harness, self.audit = build_loop(state_dir, policy, max_steps, resume=True, ask=self._ask)
        self.loop.on_proposal, self.loop.on_turn = self._show_proposal, self._show_turn

    # -- harness callbacks -------------------------------------------------------------------------

    def _ask(self, ctx: ActionContext) -> bool:
        if ctx.tool.name in self.always:
            self.write(f"    approved for this session: {ctx.tool.name}")
            return True
        self.write(f"    ! {ctx.tool.name} is {ctx.tool.risk}: {ctx.tool.description}")
        while True:
            try:
                answer = self.read("    approve? [y]es once / [a]lways this session / [n]o > ").strip().lower()
            except EOFError:
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("a", "always"):
                self.always.add(ctx.tool.name)
                return True
            if answer in ("n", "no", ""):
                return False

    def _show_proposal(self, step: int, call: ToolCall) -> None:
        self.write(f"  -> {call.tool}({_args(call.args)})")

    def _show_turn(self, turn: Turn) -> None:
        if turn.outcome == "executed":
            self.write(f"    ALLOWED  {json.dumps(turn.observation, sort_keys=True)}")
        else:
            error = turn.observation.get("error", "")
            self.write(f"    {turn.outcome.upper():<8} by {decided_by(turn.outcome, error)}: {error}")

    # -- the session -------------------------------------------------------------------------------

    def banner(self) -> str:
        spec = self.harness.spec
        return (f"Jiva harness: {spec.agent_name} ({spec.role}), acting for you\n"
                f"identity  {self.harness.citta.did}\n"
                f"ledger    {self.state_dir}/ ({len(self.harness.citta.records)} records so far)\n"
                "Type a task, or /help.")

    def run_task(self, goal: str) -> None:
        try:
            result = self.loop.run(principal="human:cli", channel="CLI", goal=goal,
                                   approved_irreversible=frozenset(self.always))
        except KeyboardInterrupt:
            self.write("\n  interrupted; the conversation continues from here")
            return
        except Exception as exc:
            message = self.recoverable(exc)
            if message is None:
                raise
            self.write(f"  {message}")
            return
        if result.stop_reason == "finished":
            self.write(f"\njiva> {result.answer}")
        else:
            self.write(f"\n  run stopped: {result.stop_reason}")
        status = ledger_status(self.harness, self.audit, self.state_dir)
        ok = status["citta"]["verified"] and status["audit"]["verified"]
        self.write(f"  [citta {status['citta']['records']} records, audit {status['audit']['entries']} entries, "
                   f"{'verified' if ok else 'VERIFICATION FAILED'}]")

    def command(self, line: str) -> bool:
        """Handle a /command. Returns False when the session should end."""
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name in ("/quit", "/exit"):
            return False
        if name == "/help":
            self.write(HELP)
        elif name == "/tools":
            for tool in TOOLS:
                scope = "in scope" if tool.capability in self.harness.spec.capabilities else "NOT in this deployment's scope"
                self.write(f"  {tool.name:<14} {tool.risk:<16} {scope}. {tool.description}")
            self.write(f"  spending limit: ${self.harness.spec.conditions['max_amount']}")
        elif name == "/approve" and arg:
            self.always.add(arg)
            self.write(f"  pre-approved: {', '.join(sorted(self.always))}")
        elif name == "/revoke" and arg:
            self.always.discard(arg)
            self.write(f"  pre-approved: {', '.join(sorted(self.always)) or 'none'}")
        elif name == "/new":
            reset = getattr(self.policy, "reset", None)
            if reset:
                reset()
            self.write("  new conversation; identity and ledger unchanged")
        elif name == "/scenarios":
            for key, scenario in SCENARIOS.items():
                self.write(f"  {key:<18} {scenario.goal}")
        elif name == "/run" and arg in SCENARIOS:
            self.write(f"  task: {SCENARIOS[arg].goal}")
            self.run_task(SCENARIOS[arg].goal)
        elif name == "/ledger":
            status = ledger_status(self.harness, self.audit, self.state_dir)
            citta, audit = status["citta"], status["audit"]
            self.write(f"  identity   {status['agent_did']}\n"
                       f"  citta      {citta['records']} records ({citta['refusals']} refusals), "
                       f"Merkle proofs verify: {citta['verified']}, root {citta['root'][:16]}...\n"
                       f"  audit log  {audit['entries']} entries, hash chain verifies: {audit['verified']}, "
                       f"one-field edit detected: {'n/a (empty log)' if audit['tamper_detected'] is None else audit['tamper_detected']}")
        else:
            self.write(f"  unknown command {line!r}; /help lists them")
        return True

    def repl(self) -> None:
        self.write(self.banner())
        while True:
            try:
                line = self.read("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.write("")
                return
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return
            else:
                self.run_task(line)
