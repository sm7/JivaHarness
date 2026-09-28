from __future__ import annotations

import argparse
import json
import sys
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


def live_demo(args: argparse.Namespace) -> int:
    from .demo import SYSTEM, render, run_demo, scripted_policy, summarize

    if args.policy == "claude":
        try:
            import anthropic
        except ImportError:
            print("live-demo --policy claude needs the Anthropic SDK: pip install -e '.[claude]'", file=sys.stderr)
            return 2
        from .claude_policy import ClaudePolicy

        policy = ClaudePolicy(model=args.model, system=SYSTEM, effort=args.effort)
        header = f"Jiva harness live demo: policy=claude model={args.model}"
    else:
        policy = scripted_policy()
        header = "Jiva harness live demo: policy=scripted (no model; replays fixed proposals)"
    approved = frozenset(args.approve)
    header += f"\nPrincipal approvals: {', '.join(sorted(approved)) or 'none'}"
    try:
        result, harness, audit = run_demo(args.state_dir, policy, approved)
    except Exception as exc:
        # the SDK raises TypeError, not an API error, when it finds no credentials at all
        no_credentials = isinstance(exc, TypeError) and "authentication" in str(exc)
        if args.policy == "claude" and (isinstance(exc, anthropic.AnthropicError) or no_credentials):
            print(f"Claude API call failed: {exc}\nSet ANTHROPIC_API_KEY, or run with --policy scripted "
                  "to see the same scenario without a model.", file=sys.stderr)
            return 2
        raise
    summary = summarize(result, harness, audit, args.state_dir)
    if args.policy == "claude":
        summary["model_usage"] = policy.usage()
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else render(summary, header))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful Jiva governance harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run one scoped actor-critic action")
    demo_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    loop_parser = subparsers.add_parser("loop-demo", help="run the multi-step governed agent loop")
    loop_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    live_parser = subparsers.add_parser(
        "live-demo", help="a model pursues a goal while the harness allows, blocks and denies its actions")
    live_parser.add_argument("--policy", choices=("claude", "scripted"), default="claude")
    live_parser.add_argument("--model", default="claude-opus-5")
    live_parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    live_parser.add_argument("--approve", action="append", default=[], metavar="TOOL",
                             help="principal approval for an irreversible tool, e.g. --approve place_order")
    live_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-live-demo"))
    live_parser.add_argument("--json", action="store_true", help="print the structured summary instead")
    args = parser.parse_args()

    if args.command == "demo":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(demo(args.state_dir), indent=2, sort_keys=True))
    elif args.command == "loop-demo":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(loop_demo(args.state_dir), indent=2, sort_keys=True))
    elif args.command == "live-demo":
        sys.exit(live_demo(args))


if __name__ == "__main__":
    main()
