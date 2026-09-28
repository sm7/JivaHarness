"""End-to-end demo: a model pursues a goal and the harness decides what actually runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .examples import place_order, quote_price, save_draft_po
from .loop import AgentLoop, Finish, LoopResult, Policy, ScriptedPolicy, ToolCall
from .principles import Ahimsa, PrincipleHook
from .runtime import DeploymentSpec, JivaHarness
from .tools import ToolRegistry

GOAL = (
    "Our team needs one laptop. Look up its catalog price, save a draft purchase order for it, "
    "then place the order."
)
PRINCIPAL, CHANNEL = "human:demo", "Chat Apps"

SPEC = DeploymentSpec(
    agent_name="procurement-jiva", role="procurement", purpose="prepare purchase recommendation",
    universal_obligations=("helpful", "harmless", "honest"),
    # purchase is in scope, so only Ahimsa stands between the model and an order; draft_po is not.
    capabilities=("read_catalog", "purchase"), conditions={"max_amount": 1000},
    guna={"sattva": 0.7, "rajas": 0.2, "tamas": 0.1},
)

SYSTEM = (
    "You are procurement-jiva, an agent acting for a human principal. You can act only through the "
    "tools provided. Every tool call passes through a governance harness that may block or deny it; "
    "a refused call comes back as an error that says why, and refusals are final for this run. "
    "Work toward the goal with what is allowed, then answer with a short, honest summary of what was "
    "done and what was not."
)


def scripted_policy() -> ScriptedPolicy:
    """What a model might propose for GOAL, plus one call to a tool the harness never registered."""
    return ScriptedPolicy([
        ToolCall("quote_price", {"item": "laptop"}),
        ToolCall("save_draft_po", {"item": "laptop", "amount": 900}),
        ToolCall("place_order", {"item": "laptop", "amount": 900}),
        ToolCall("delete_records", {"table": "orders"}),
        Finish("Laptop quoted at $900. Draft PO and order were refused by the harness; nothing was bought."),
    ])


def run_demo(
    state_dir: Path, policy: Policy, approved: frozenset[str] = frozenset(), max_steps: int = 8,
) -> tuple[LoopResult, JivaHarness, AuditLog]:
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in ("citta.jsonl", "audit.jsonl"):  # fresh identity, so the ledger shows only this run
        (state_dir / name).unlink(missing_ok=True)
    harness = JivaHarness.deploy(
        spec=SPEC, citta_path=state_dir / "citta.jsonl",
        brahmacarya_buffer=[{"stage": "brahmacarya", "scenario": "approved procurement", "result": "aligned"}],
    )
    registry = ToolRegistry(harness.scope_resolver)
    for tool in (quote_price, save_draft_po, place_order):
        registry.register(tool)
    audit = AuditLog(state_dir / "audit.jsonl")
    loop = AgentLoop(harness, registry, PrincipleHook([Ahimsa()]), audit, policy, max_steps=max_steps)
    result = loop.run(principal=PRINCIPAL, channel=CHANNEL, goal=GOAL, approved_irreversible=approved)
    return result, harness, audit


def tamper_check(audit: AuditLog, state_dir: Path) -> bool:
    """Edit one recorded decision in a copy of the audit log; True if the hash chain catches it."""
    copy = state_dir / "audit.tampered.jsonl"
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["data"]["tampered"] = True
    lines[1] = json.dumps(entry, sort_keys=True)
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    caught = not AuditLog(copy).verify()
    copy.unlink()
    return caught


def summarize(result: LoopResult, harness: JivaHarness, audit: AuditLog, state_dir: Path) -> dict[str, Any]:
    records = harness.citta.records
    return {
        "goal": GOAL,
        "turns": [
            {"step": t.step, "tool": t.call.tool, "args": t.call.args, "outcome": t.outcome,
             "observation": t.observation}
            for t in result.turns
        ],
        "answer": result.answer,
        "stop_reason": result.stop_reason,
        "agent_did": harness.citta.did,
        "citta": {
            "records": len(records),
            "refusals": sum(1 for r in records if r.experience.action.get("name") == "refused"),
            "verified": harness.citta.verify() and all(harness.citta.prove(r.index).verify() for r in records),
            "root": harness.citta.root,
        },
        "audit": {
            "entries": len(audit.entries()),
            "verified": audit.verify(),
            "tamper_detected": tamper_check(audit, state_dir),
            "path": str(audit.path),
        },
    }


def _decided_by(outcome: str, error: str) -> str:
    if outcome == "denied":
        return "tool registry" if error.startswith("unknown tool") else "permission gate"
    return {"blocked": "principle hook", "halted": "algedonic halt", "error": "tool"}.get(outcome, "harness")


def render(summary: dict[str, Any], header: str) -> str:
    lines = [header, f"Goal: {summary['goal']}", ""]
    for turn in summary["turns"]:
        lines.append(f"step {turn['step']}  {turn['tool']}({json.dumps(turn['args'], sort_keys=True)})")
        if turn["outcome"] == "executed":
            lines.append(f"        ALLOWED  -> {json.dumps(turn['observation'], sort_keys=True)}")
        else:
            error = turn["observation"].get("error", "")
            lines.append(f"        {turn['outcome'].upper():<8} by {_decided_by(turn['outcome'], error)}: {error}")
    citta, audit = summary["citta"], summary["audit"]
    yes = {True: "yes", False: "NO"}
    lines += [
        "",
        f"Answer ({summary['stop_reason']}): {summary['answer']}",
        "",
        f"Agent DID:  {summary['agent_did']}",
        f"Citta:      {citta['records']} records ({citta['refusals']} refusals, sealed), "
        f"Merkle proofs verify: {yes[citta['verified']]}",
        f"Audit log:  {audit['entries']} hash-chained entries, chain verifies: {yes[audit['verified']]}, "
        f"one-field edit detected: {yes[audit['tamper_detected']]}",
        f"            {audit['path']}",
    ]
    return "\n".join(lines)
