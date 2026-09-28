from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .audit import AuditLog
from .demo import SCENARIOS
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


NO_KEY = "Set ANTHROPIC_API_KEY_JIVA (or ANTHROPIC_API_KEY)."


def _claude_policy(args: argparse.Namespace, keep_history: bool = False):
    """ClaudePolicy plus the SDK's error base class, or (None, None) after printing why not."""
    try:
        import anthropic
    except ImportError:
        print("--policy claude needs the Anthropic SDK: pip install -e '.[claude]'", file=sys.stderr)
        return None, None
    if not args.verbose:  # the SDK and its HTTP client can log every request at DEBUG
        for name in ("anthropic", "httpx", "httpx2", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)
    from .claude_policy import ClaudePolicy
    from .demo import SYSTEM

    policy = ClaudePolicy(model=args.model, system=SYSTEM, effort=args.effort, keep_history=keep_history)
    return policy, anthropic.AnthropicError


def _api_failure(exc: Exception, api_error: type | None) -> bool:
    # the SDK raises TypeError, not an API error, when it finds no credentials at all
    no_credentials = isinstance(exc, TypeError) and "authentication" in str(exc)
    return api_error is not None and (isinstance(exc, api_error) or no_credentials)


def live_demo(args: argparse.Namespace) -> int:
    from .demo import SCENARIOS, render, run_demo, scripted_policy, summarize

    if args.list_scenarios:
        for name, scenario in SCENARIOS.items():
            approve = f" [approves {', '.join(sorted(scenario.approve))}]" if scenario.approve else ""
            print(f"{name}{approve}\n  goal:   {scenario.goal}\n  expect: {scenario.expect}")
        return 0
    if args.goal and args.policy == "scripted":
        print("--goal needs --policy claude: the scripted policy only knows the ready-made scenarios.", file=sys.stderr)
        return 2
    scenario = SCENARIOS[args.scenario]
    goal = args.goal or scenario.goal
    approved = frozenset(args.approve) | (frozenset() if args.goal else scenario.approve)
    api_error = None
    if args.policy == "claude":
        policy, api_error = _claude_policy(args)
        if policy is None:
            return 2
        header = f"Jiva harness live demo: policy=claude model={args.model}"
    else:
        policy = scripted_policy(args.scenario)
        header = "Jiva harness live demo: policy=scripted (no model; replays fixed proposals)"
    header += f"\nScenario: {'custom goal' if args.goal else args.scenario}"
    header += f"\nPrincipal approvals: {', '.join(sorted(approved)) or 'none'}"
    try:
        result, harness, audit = run_demo(args.state_dir, policy, approved, goal=goal)
    except Exception as exc:
        if _api_failure(exc, api_error):
            print(f"Claude API call failed: {exc}\n{NO_KEY} Or use --policy scripted to run without a model.",
                  file=sys.stderr)
            return 2
        raise
    summary = summarize(result, harness, audit, args.state_dir, goal)
    if args.policy == "claude":
        summary["model_usage"] = policy.usage()
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else render(summary, header))
    return 0


def session_command(args: argparse.Namespace) -> int:
    from .session import Session

    policy, api_error = _claude_policy(args, keep_history=True)
    if policy is None:
        return 2

    def recoverable(exc: Exception) -> str | None:
        return f"Claude API call failed: {exc}\n  {NO_KEY}" if _api_failure(exc, api_error) else None

    Session(policy, args.state_dir, recoverable=recoverable).repl()
    return 0


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--verbose", action="store_true", help="keep the Anthropic SDK's debug logging")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jiva governance harness. With no command, starts an interactive session: you give the "
                    "agent tasks, and every action it proposes is judged by the harness as it happens.")
    _add_model_args(parser)
    parser.add_argument("--state-dir", type=Path, default=Path(".jiva"),
                        help="where the agent's identity ledger and audit log live (kept between sessions)")
    subparsers = parser.add_subparsers(dest="command")
    demo_parser = subparsers.add_parser("demo", help="run one scoped actor-critic action")
    demo_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    loop_parser = subparsers.add_parser("loop-demo", help="run the multi-step governed agent loop")
    loop_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-state"))
    live_parser = subparsers.add_parser(
        "live-demo", help="a model pursues a goal while the harness allows, blocks and denies its actions")
    live_parser.add_argument("--policy", choices=("claude", "scripted"), default="claude")
    _add_model_args(live_parser)
    live_parser.add_argument("--scenario", default="procurement", choices=sorted(SCENARIOS),
                             help="a ready-made goal; --list-scenarios shows what each tests")
    live_parser.add_argument("--goal", help="your own goal for the agent (needs --policy claude)")
    live_parser.add_argument("--list-scenarios", action="store_true")
    live_parser.add_argument("--approve", action="append", default=[], metavar="TOOL",
                             help="principal approval for an irreversible tool, e.g. --approve place_order")
    live_parser.add_argument("--state-dir", type=Path, default=Path(".jiva-live-demo"))
    live_parser.add_argument("--json", action="store_true", help="print the structured summary instead")
    args = parser.parse_args()

    if args.command is None:
        sys.exit(session_command(args))
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
