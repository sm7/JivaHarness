import copy
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from jiva_harness.claude_policy import ClaudePolicy
from jiva_harness.demo import SCENARIOS, build_loop, chat, run_demo, scripted_policy, summarize
from jiva_harness.loop import Finish


def tool_use(id_, name, **input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def reply(*blocks, stop_reason="tool_use"):
    return NS(content=list(blocks), stop_reason=stop_reason, stop_details=None,
              usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """Stands in for anthropic.Anthropic(): replays responses and snapshots every request."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return self.responses[len(self.requests) - 1]


def scenario_client():
    return FakeClient([
        reply(NS(type="thinking", thinking="", signature="sig"), tool_use("t1", "quote_price", item="laptop")),
        reply(tool_use("t2", "save_draft_po", item="laptop", amount=900)),
        reply(tool_use("t3", "place_order", item="laptop", amount=900)),
        reply(NS(type="text", text="Quoted $900; draft PO and order were refused."), stop_reason="end_turn"),
    ])


def test_claude_policy_drives_the_governed_loop(tmp_path):
    client = scenario_client()

    result, harness, audit = run_demo(tmp_path, ClaudePolicy(client=client, system="sys"))

    assert [(t.call.tool, t.outcome) for t in result.turns] == [
        ("quote_price", "executed"), ("save_draft_po", "denied"), ("place_order", "blocked")]
    assert result.stop_reason == "finished" and result.answer.startswith("Quoted $900")
    summary = summarize(result, harness, audit, tmp_path)
    assert summary["citta"] == {**summary["citta"], "records": 3, "refusals": 2, "verified": True}
    assert summary["audit"]["verified"] and summary["audit"]["tamper_detected"]


def test_claude_sees_each_verdict_as_the_tool_result_for_its_own_call(tmp_path):
    client = scenario_client()

    run_demo(tmp_path, ClaudePolicy(client=client))

    results = [m["content"][0] for m in client.requests[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert [r["is_error"] for r in results] == [False, True, True]
    assert json.loads(results[0]["content"]) == {"outcome": "executed", "item": "laptop", "amount": 900}
    assert json.loads(results[2]["content"])["outcome"] == "blocked"
    assert "ahimsa" in json.loads(results[2]["content"])["error"]


def test_request_shape_and_append_only_transcript(tmp_path):
    client = scenario_client()

    run_demo(tmp_path, ClaudePolicy(client=client, system="sys"))

    first, last = client.requests[0], client.requests[-1]
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert first["thinking"] == {"type": "adaptive"} and first["system"] == "sys"
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    tools = {t["name"]: t for t in first["tools"]}
    assert set(tools) == {"quote_price", "save_draft_po", "place_order"}
    assert tools["place_order"]["input_schema"]["required"] == ["item", "amount"]
    assert "external_effect" in tools["place_order"]["description"]
    # every earlier request is a prefix of the next, and thinking blocks go back unchanged
    for earlier, later in zip(client.requests, client.requests[1:]):
        assert later["messages"][:len(earlier["messages"])] == earlier["messages"]
    assert last["messages"][1]["content"][0].type == "thinking"


def test_principal_approval_lets_the_order_through(tmp_path):
    result, harness, _ = run_demo(tmp_path, ClaudePolicy(client=scenario_client()), approved=frozenset({"place_order"}))

    assert [t.outcome for t in result.turns] == ["executed", "denied", "executed"]
    assert result.turns[2].observation["status"] == "submitted"


def test_tool_the_harness_never_offered_is_denied(tmp_path):
    client = FakeClient([
        reply(tool_use("t1", "delete_records", table="orders")),
        reply(NS(type="text", text="done"), stop_reason="end_turn"),
    ])

    result, harness, _ = run_demo(tmp_path, ClaudePolicy(client=client))

    assert result.turns[0].outcome == "denied"
    assert result.turns[0].observation["error"] == "unknown tool: delete_records"


def test_model_refusal_and_truncation_end_the_run():
    refused = reply(stop_reason="refusal")
    refused.stop_details = NS(explanation="declined")
    policy = ClaudePolicy(client=FakeClient([refused]))
    assert policy.propose("g", [], []) == Finish("model refused: declined")

    truncated = ClaudePolicy(client=FakeClient([reply(tool_use("t", "quote_price"), stop_reason="max_tokens")]))
    assert "max_tokens" in truncated.propose("g", [], []).answer


def test_policy_resets_between_runs(tmp_path):
    client = FakeClient([
        reply(tool_use("a", "quote_price", item="laptop")), reply(NS(type="text", text="one"), stop_reason="end_turn"),
        reply(NS(type="text", text="two"), stop_reason="end_turn"),
    ])
    policy = ClaudePolicy(client=client)

    run_demo(tmp_path, policy)
    assert policy.propose("new goal", [], []) == Finish("two")
    assert client.requests[-1]["messages"] == [{"role": "user", "content": "new goal"}]
    assert policy.usage() == {"requests": 1, "input_tokens": 10, "output_tokens": 5}


def test_cli_live_demo_scripted(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "jiva_harness.cli", "live-demo", "--policy", "scripted",
         "--state-dir", str(tmp_path), "--json"],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(completed.stdout)
    assert [t["outcome"] for t in summary["turns"]] == ["executed", "denied", "blocked", "denied"]
    assert summary["citta"]["verified"] and summary["audit"]["verified"] and summary["audit"]["tamper_detected"]


def test_api_key_prefers_jiva_variable(monkeypatch):
    from jiva_harness.claude_policy import api_key_from_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "general")
    monkeypatch.setenv("ANTHROPIC_API_KEY_JIVA", "jiva")
    assert api_key_from_env() == "jiva"
    assert ClaudePolicy(fallbacks=False).client.api_key == "jiva"
    monkeypatch.delenv("ANTHROPIC_API_KEY_JIVA")
    assert api_key_from_env() == "general"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert api_key_from_env() is None


EXPECTED = {
    "procurement": ["executed", "denied", "blocked", "denied"],
    "harmful": ["blocked"],
    "unknown-tool": ["denied", "denied"],
    "approved-purchase": ["executed", "executed"],
    "over-budget": ["executed", "denied"],
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_each_scenario_gets_the_verdicts_it_describes(tmp_path, name):
    scenario = SCENARIOS[name]
    result, harness, audit = run_demo(tmp_path, scripted_policy(name), scenario.approve, goal=scenario.goal)

    assert [t.outcome for t in result.turns] == EXPECTED[name]
    assert harness.citta.verify() and audit.verify()


def test_custom_goal_reaches_claude_and_the_ledger(tmp_path):
    client = FakeClient([reply(NS(type="text", text="ok"), stop_reason="end_turn")])

    result, harness, audit = run_demo(tmp_path, ClaudePolicy(client=client), goal="find me a chair")

    assert client.requests[0]["messages"][0] == {"role": "user", "content": "find me a chair"}
    assert audit.entries()[0]["data"]["goal"] == "find me a chair"
    assert summarize(result, harness, audit, tmp_path, "find me a chair")["goal"] == "find me a chair"


def test_chat_runs_goals_with_approvals_on_one_ledger(tmp_path):
    client = FakeClient([
        reply(tool_use("a", "place_order", item="laptop", amount=900)),
        reply(NS(type="text", text="blocked"), stop_reason="end_turn"),
        reply(tool_use("b", "place_order", item="laptop", amount=900)),
        reply(NS(type="text", text="ordered"), stop_reason="end_turn"),
    ])
    loop, harness, audit = build_loop(tmp_path, ClaudePolicy(client=client))
    lines = iter(["buy a laptop", "/approve place_order", "buy a laptop", "/bogus", "/quit"])
    out: list[str] = []

    chat(loop, harness, audit, tmp_path, read=lambda _: next(lines), write=out.append)

    text = "\n".join(out)
    assert "BLOCKED  by principle hook" in text and "ALLOWED  ->" in text
    assert "unknown command /bogus" in text
    assert [r.experience.action["name"] for r in harness.citta.records] == ["refused", "purchase"]
    assert audit.verify() and [e["event"] for e in audit.entries()].count("run_started") == 2


def test_chat_ends_on_eof(tmp_path):
    loop, harness, audit = build_loop(tmp_path, ClaudePolicy(client=FakeClient([])))

    def eof(_):
        raise EOFError

    chat(loop, harness, audit, tmp_path, read=eof, write=lambda _: None)
    assert audit.entries() == []


def test_cli_rejects_custom_goal_without_a_model(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "jiva_harness.cli", "live-demo", "--policy", "scripted", "--goal", "x",
         "--state-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2 and "--goal needs --policy claude" in completed.stderr


def test_sdk_debug_logging_is_quiet_unless_verbose(monkeypatch):
    import argparse
    import logging

    from jiva_harness.cli import _claude_policy

    monkeypatch.setenv("ANTHROPIC_API_KEY_JIVA", "k")
    logging.getLogger("anthropic").setLevel(logging.DEBUG)
    _claude_policy(argparse.Namespace(model="m", effort="low", verbose=True))
    assert logging.getLogger("anthropic").level == logging.DEBUG
    _claude_policy(argparse.Namespace(model="m", effort="low", verbose=False))
    assert logging.getLogger("anthropic").level == logging.WARNING
