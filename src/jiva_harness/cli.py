from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditLog
from .examples import place_order, quote_price
from .loop import AgentLoop, Finish, ScriptedPolicy, ToolCall
from .principles import Ahimsa, PrincipleHook
from .runtime import DeploymentSpec, JivaHarness, PrincipalRequest
from .tools import ToolRegistry


def demo(state_dir: Path) -> dict[str, object]:
    harness = JivaHarness.deploy(
        spec=DeploymentSpec(
            agent_name="procurement-jiva",
            role="procurement",
            purpose="prepare purchase recommendation",
            universal_obligations=("helpful", "harmless", "honest"),
            capabilities=("recommend",),
            conditions={"max_amount": 1000},
            guna={"sattva": 0.7, "rajas": 0.2, "tamas": 0.1},
        ),
        citta_path=state_dir / "citta.jsonl",
        brahmacarya_buffer=[
            {"stage": "brahmacarya", "scenario": "approved procurement", "result": "aligned"}
        ],
    )
    result = harness.run(PrincipalRequest(
        principal="human:demo",
        channel="Chat Apps",
        goal="recommend a laptop",
        observation={"amount": 900},
        requested_capability="recommend",
    ))
    record = harness.citta.records[-1]
    proof = harness.citta.prove(record.index)
    return {
        "authorized": result.authorized,
        "agent_did": result.agent_did,
        "genesis_hash": harness.citta.genesis_hash,
        "citta_root": result.citta_root,
        "citta_verified": harness.citta.verify() and proof.verify(),
        "credential": {
            "scope": list(result.credential.scope),
            "capability": result.credential.capability,
            "role": result.credential.role,
            "conditions": result.credential.conditions,
            "ephemeral": result.credential.ephemeral,
            "transferable": result.credential.transferable,
            "identity_scoped": result.credential.identity_scoped,
        },
        "action": result.action,
        "reward": result.reward.to_dict(),
        "paper_tuple_fields": list(record.experience.to_dict()),
    }


def loop_demo(state_dir: Path) -> dict[str, object]:
    harness = JivaHarness.deploy(
        spec=DeploymentSpec(
            agent_name="procurement-jiva",
            role="procurement",
            purpose="prepare purchase recommendation",
            universal_obligations=("helpful", "harmless", "honest"),
            capabilities=("read_catalog",),
            conditions={"max_amount": 1000},
            guna={"sattva": 0.7, "rajas": 0.2, "tamas": 0.1},
        ),
        citta_path=state_dir / "citta.jsonl",
        brahmacarya_buffer=[
            {"stage": "brahmacarya", "scenario": "approved procurement", "result": "aligned"}
        ],
    )
    registry = ToolRegistry(harness.scope_resolver)
    registry.register(quote_price)
    registry.register(place_order)
    audit = AuditLog(state_dir / "audit.jsonl")
    policy = ScriptedPolicy([
        ToolCall("quote_price", {"item": "laptop"}),
        ToolCall("place_order", {"item": "laptop", "amount": 900}),
        ToolCall("delete_records", {}),
        Finish("laptop quoted at 900; order not placed without principal approval"),
    ])
    result = AgentLoop(harness, registry, PrincipleHook([Ahimsa()]), audit, policy).run(
        principal="human:demo", channel="Chat Apps", goal="recommend a laptop",
    )
    return {
        "answer": result.answer,
        "stop_reason": result.stop_reason,
        "turns": [
            {"step": t.step, "tool": t.call.tool, "outcome": t.outcome, "observation": t.observation}
            for t in result.turns
        ],
        "agent_did": harness.citta.did,
        "citta_records": len(harness.citta.records),
        "citta_verified": harness.citta.verify(),
        "audit_entries": len(audit.entries()),
        "audit_verified": audit.verify(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful Jiva governance harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run one scoped actor-critic action")
    demo_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    loop_parser = subparsers.add_parser("loop-demo", help="run the multi-step governed agent loop")
    loop_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    args = parser.parse_args()

    if args.command == "demo":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(demo(args.state_dir), indent=2, sort_keys=True))
    elif args.command == "loop-demo":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(loop_demo(args.state_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
