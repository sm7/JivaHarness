import json
import subprocess
import sys

import pytest

from jiva_harness.audit import AuditLog
from jiva_harness.examples import place_order, quote_price
from jiva_harness.loop import AgentLoop, Finish, ScriptedPolicy, ToolCall
from jiva_harness.principles import ActionContext, Ahimsa, PrincipleHook, Verdict
from jiva_harness.runtime import DeploymentSpec, JivaHarness
from jiva_harness.sealing import PlaintextSealer
from jiva_harness.tools import Tool, ToolRegistry


def build(tmp_path, proposals, capabilities=("read_catalog",), max_steps=8, principles=None):
    harness = JivaHarness.deploy(
        spec=DeploymentSpec(
            agent_name="procurement-jiva", role="procurement",
            purpose="prepare purchase recommendation",
            universal_obligations=("helpful", "harmless", "honest"),
            capabilities=capabilities, conditions={"max_amount": 1000},
            guna={"sattva": 0.7, "rajas": 0.2, "tamas": 0.1},
        ),
        citta_path=tmp_path / "citta.jsonl",
        brahmacarya_buffer=[{"scenario": "approved procurement"}],
    )
    registry = ToolRegistry(harness.scope_resolver)
    registry.register(quote_price)
    registry.register(place_order)
    audit = AuditLog(tmp_path / "audit.jsonl", clock=lambda: "t")
    loop = AgentLoop(harness, registry, PrincipleHook(principles or [Ahimsa()]), audit, ScriptedPolicy(proposals), max_steps)
    return harness, audit, loop


def run(loop, **kwargs):
    return loop.run(principal="human:alice", channel="Chat Apps", goal="recommend a laptop", **kwargs)


def test_allowed_tool_executes_and_is_recorded_in_citta_and_audit(tmp_path):
    harness, audit, loop = build(tmp_path, [ToolCall("quote_price", {"item": "laptop"}), Finish("900")])

    result = run(loop)

    assert result.stop_reason == "finished" and result.answer == "900"
    assert [t.outcome for t in result.turns] == ["executed"]
    assert result.turns[0].observation == {"item": "laptop", "amount": 900}
    assert len(harness.citta.records) == 1 and harness.citta.verify()
    assert harness.citta.records[0].experience.action["tool"] == "quote_price"
    events = [e["event"] for e in audit.entries()]
    assert events == ["run_started", "proposal", "principle_check", "credential_issued", "tool_executed", "run_finished"]
    assert audit.verify()


def test_principle_hook_blocks_irreversible_tool_before_permission_gate(tmp_path):
    # purpose granted, so only Ahimsa stands between the model and the order
    harness, audit, loop = build(
        tmp_path, [ToolCall("place_order", {"item": "laptop", "amount": 900})],
        capabilities=("read_catalog", "purchase"),
    )

    result = run(loop)

    assert result.turns[0].outcome == "blocked"
    assert "ahimsa" in result.turns[0].observation["error"]
    assert [r.experience.action["outcome"] for r in harness.citta.records] == ["blocked"]
    assert "credential_issued" not in [e["event"] for e in audit.entries()]


def test_principal_approval_unblocks_irreversible_tool(tmp_path):
    harness, _, loop = build(
        tmp_path, [ToolCall("place_order", {"item": "laptop", "amount": 900})],
        capabilities=("read_catalog", "purchase"),
    )

    result = run(loop, approved_irreversible=frozenset({"place_order"}))

    assert result.turns[0].outcome == "executed"
    assert len(harness.citta.records) == 1


def test_permission_gate_denies_ungranted_capability_and_condition_breach(tmp_path):
    harness, _, loop = build(tmp_path, [
        ToolCall("place_order", {"item": "laptop", "amount": 900}),   # capability not granted
        ToolCall("quote_price", {"item": "laptop", "amount": 5000}),  # breaches max_amount
    ])

    result = run(loop, approved_irreversible=frozenset({"place_order"}))

    assert [t.outcome for t in result.turns] == ["denied", "denied"]
    assert [r.experience.action["outcome"] for r in harness.citta.records] == ["denied", "denied"]


def test_unknown_tool_is_denied_and_tool_errors_become_observations(tmp_path):
    harness, audit, loop = build(tmp_path, [
        ToolCall("delete_records", {}),
        ToolCall("quote_price", {"item": "yacht"}),
    ])

    result = run(loop)

    assert [t.outcome for t in result.turns] == ["denied", "error"]
    assert "tool_error" in [e["event"] for e in audit.entries()]


def test_loop_stops_at_max_steps(tmp_path):
    _, audit, loop = build(tmp_path, [ToolCall("quote_price", {"item": "laptop"})] * 5, max_steps=3)

    result = run(loop)

    assert result.stop_reason == "max_steps" and result.answer is None
    assert len(result.turns) == 3
    assert audit.entries()[-1]["event"] == "run_stopped"


def test_audit_log_detects_tampering_and_resumes_chain(tmp_path):
    _, audit, loop = build(tmp_path, [ToolCall("quote_price", {"item": "laptop"}), Finish("ok")])
    run(loop)
    assert audit.verify()

    reopened = AuditLog(tmp_path / "audit.jsonl", clock=lambda: "t")
    reopened.record("note", "did:jiva:x")
    assert reopened.verify()

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    entry = json.loads(lines[1])
    entry["data"]["args"] = {"item": "server"}
    lines[1] = json.dumps(entry, sort_keys=True)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    assert not reopened.verify()


class HaltOnServer:
    name = "tripwire"

    def check(self, ctx: ActionContext) -> Verdict:
        if ctx.args.get("item") == "server":
            return Verdict(self.name, False, "server purchases need S5 review", halt=True)
        return Verdict(self.name, True, "ok")


def test_principle_can_raise_algedonic_halt_that_stops_the_run(tmp_path):
    harness, audit, loop = build(
        tmp_path,
        [ToolCall("quote_price", {"item": "server"}), ToolCall("quote_price", {"item": "laptop"}), Finish("x")],
        principles=[Ahimsa(), HaltOnServer()],
    )

    result = run(loop)

    assert result.stop_reason == "halted" and result.answer is None
    assert [t.outcome for t in result.turns] == ["halted"]
    assert [r.experience.action["outcome"] for r in harness.citta.records] == ["halted"]
    halt = audit.entries()[-1]
    assert halt["event"] == "algedonic_halt" and halt["data"]["raised_by"] == "tripwire"


def test_external_halt_stops_before_next_proposal(tmp_path):
    class HaltAfterFirst(ScriptedPolicy):
        loop = None

        def propose(self, goal, history, tools):
            if history:
                self.loop.halt("human:alice", "stop")
            return super().propose(goal, history, tools)

    policy = HaltAfterFirst([ToolCall("quote_price", {"item": "laptop"}), ToolCall("quote_price", {"item": "monitor"})])
    harness, audit, loop = build(tmp_path, [])
    loop.policy, policy.loop = policy, loop
    loop.max_steps = 3

    result = loop.run(principal="human:alice", channel="Chat Apps", goal="g")

    # halt raised while proposing step 2 takes effect before that proposal executes
    assert result.stop_reason == "halted"
    assert [t.call.args["item"] for t in result.turns] == ["laptop"]


def test_refusals_are_citta_leaves_with_details_only_inside_the_sealed_envelope(tmp_path):
    harness, audit, loop = build(tmp_path, [
        ToolCall("quote_price", {"item": "laptop"}),
        ToolCall("place_order", {"item": "laptop", "amount": 900}),
        ToolCall("delete_records", {"table": "users"}),
    ])

    run(loop)

    refused = [r for r in harness.citta.records if r.experience.action["name"] == "refused"]
    assert [r.experience.action["outcome"] for r in refused] == ["blocked", "denied"]
    for record in refused:
        exp = record.experience.to_dict()
        assert set(exp) == {"state", "goal", "constraints", "action", "reward", "activation_trace"}
        assert exp["reward"]["evaluated"] is False
        assert "partial_observation" not in exp["state"]
        outside = {k: v for k, v in exp["action"].items() if k != "sealed"}
        assert "users" not in json.dumps(outside) and "users" not in json.dumps(exp["state"])
        assert harness.citta.prove(record.index).verify()
    assert PlaintextSealer().unseal(refused[1].experience.action["sealed"]) == {
        "tool": "delete_records", "args": {"table": "users"}, "reason": "unknown tool: delete_records",
    }
    assert harness.citta.verify()
    assert [e["event"] for e in audit.entries()].count("refusal_recorded") == 2


def test_custom_sealer_output_is_what_citta_stores_and_hashes(tmp_path):
    class RedactingSealer:
        scheme = "test-redact"

        def seal(self, payload):
            return {"scheme": self.scheme, "ciphertext": "opaque"}

        def unseal(self, envelope):
            raise NotImplementedError

    harness, _, loop = build(tmp_path, [ToolCall("delete_records", {"table": "users"})])
    loop.sealer = RedactingSealer()

    run(loop)

    record = harness.citta.records[0]
    assert record.experience.action["sealed"] == {"scheme": "test-redact", "ciphertext": "opaque"}
    assert "users" not in (tmp_path / "citta.jsonl").read_text()
    assert harness.citta.verify() and harness.citta.prove(record.index).verify()


def test_duplicate_tool_registration_is_rejected():
    registry = ToolRegistry()
    registry.register(quote_price)
    with pytest.raises(ValueError):
        registry.register(Tool("quote_price", "x", "dup", lambda a: {}))


def test_cli_loop_demo(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "jiva_harness.cli", "loop-demo", "--state-dir", str(tmp_path)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert [t["outcome"] for t in output["turns"]] == ["executed", "blocked", "denied"]
    assert output["citta_verified"] and output["audit_verified"]
