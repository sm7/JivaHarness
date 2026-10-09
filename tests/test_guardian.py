import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jiva_harness.demo import build_loop
from jiva_harness.examples import place_order, quote_price
from jiva_harness.guardian import (
    HARM, FakeHarmDetector, GraniteGuardian, GuardianUnavailable, judged_messages, parse_score,
)
from jiva_harness.loop import Finish, ScriptedPolicy, ToolCall
from jiva_harness.principles import ActionContext, HarmCheck
from jiva_harness.runtime import Reward


def ctx(tool, args=None):
    return ActionContext("buy a laptop", "human:alice", tool, args or {}, 1)


def reply(content):
    return {"choices": [{"message": {"content": content}}]}


def test_parse_score_takes_the_last_score_and_ignores_think():
    assert parse_score("<think>\n</think><score> Yes </score>") == "yes"
    assert parse_score("<score>no</score> ... <score>yes</score>") == "yes"
    assert parse_score("no tags") is None


def test_prompt_puts_goal_call_and_guardian_block_in_order():
    messages = judged_messages("buy a laptop", "place_order", {"amount": 900}, {"status": "submitted"}, HARM)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert json.loads(messages[1]["content"].split("\nResult: ")[0]) == [
        {"name": "place_order", "arguments": {"amount": 900}}]
    assert messages[1]["content"].endswith('Result: {"status": "submitted"}')
    block = messages[2]["content"]
    assert block.startswith("<guardian><no-think>") and f"### Criteria: {HARM}" in block
    assert "### Scoring Schema: If the last assistant's text meets the criteria" in block


def test_granite_guardian_maps_yes_to_harmful_and_sends_a_deterministic_request():
    sent = []

    def post(url, body, timeout):
        sent.append((url, body))
        return reply("<think>\n</think>\n<score>yes</score>")

    guardian = GraniteGuardian("http://127.0.0.1:8080/v1/", post=post)
    assessment = guardian.assess("goal", "place_order", {"amount": 900})

    assert assessment.harmful and assessment.model == "ibm-granite/granite-guardian-4.1-8b"
    url, body = sent[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert body["temperature"] == 0.0 and body["messages"][-1]["content"].startswith("<guardian>")


@pytest.mark.parametrize("post", [
    lambda url, body, timeout: reply("I think it's fine"),
    lambda url, body, timeout: (_ for _ in ()).throw(ConnectionRefusedError()),
])
def test_no_verdict_raises_guardian_unavailable(post):
    with pytest.raises(GuardianUnavailable):
        GraniteGuardian("http://x/v1", post=post).assess("goal", "quote_price", {})


def test_granite_guardian_over_real_http():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            harmful = "delete" in body["messages"][1]["content"]
            payload = json.dumps(reply(f"<score>{'yes' if harmful else 'no'}</score>")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        guardian = GraniteGuardian(f"http://127.0.0.1:{server.server_port}/v1", timeout=5)
        assert guardian.assess("goal", "delete_records", {}).harmful
        assert not guardian.assess("goal", "quote_price", {"item": "laptop"}).harmful
    finally:
        server.shutdown()


def test_harm_check_blocks_harmful_calls_and_never_approves():
    check = HarmCheck(FakeHarmDetector(harmful_terms=("laptop",)))
    assert not check.check(ctx(quote_price, {"item": "laptop"})).allowed
    assert check.check(ctx(quote_price, {"item": "monitor"})).allowed


def test_harm_check_fails_closed_only_for_consequential_tools():
    check = HarmCheck(FakeHarmDetector(fail=True))
    assert not check.check(ctx(place_order, {"amount": 900})).allowed
    verdict = check.check(ctx(quote_price, {"item": "laptop"}))
    assert verdict.allowed and "unavailable" in verdict.reason


def test_harm_term_only_ever_lowers_the_reward():
    base = Reward(task=1.0, sreya=1.0, sattva_divergence=0.3)
    assert Reward(1.0, 1.0, 0.3, harm=0.0).total == base.total
    assert Reward(1.0, 1.0, 0.3, harm=1.0).total == base.total - 1.0


def test_loop_with_guardian_blocks_before_execution_and_scores_after(tmp_path):
    detector = FakeHarmDetector(harmful_terms=("server",))
    policy = ScriptedPolicy([
        ToolCall("quote_price", {"item": "laptop"}),
        ToolCall("quote_price", {"item": "server"}),
        Finish("done"),
    ])
    loop, harness, audit = build_loop(tmp_path, policy, guardian=detector)

    result = loop.run(principal="human:alice", channel="Chat Apps", goal="price things")

    assert [t.outcome for t in result.turns] == ["executed", "blocked"]
    assert "harm_check" in result.turns[1].observation["error"]
    # pre-check on both proposals, post-check only on the one that ran (with its observation)
    assert [(tool, obs is not None) for tool, _, obs in detector.calls] == [
        ("quote_price", False), ("quote_price", True), ("quote_price", False)]
    executed = [e for e in audit.entries() if e["event"] == "tool_executed"]
    assert executed[0]["data"]["reward"]["harm"] == 0.0
    assert [r.experience.action.get("outcome", r.experience.action["name"]) for r in harness.citta.records] == [
        "blocked", "rollup"]


def test_harm_flag_after_execution_keeps_the_action_in_citta(tmp_path):
    class FlagOutcomes(FakeHarmDetector):
        def assess(self, goal, tool, args, observation=None):
            result = super().assess(goal, tool, args, observation)
            return type(result)(observation is not None, result.criterion, "fake", "")

    loop, harness, _ = build_loop(tmp_path, ScriptedPolicy([ToolCall("quote_price", {"item": "laptop"}), Finish("ok")]),
                                  guardian=FlagOutcomes())
    loop.run(principal="human:alice", channel="Chat Apps", goal="price a laptop")

    [record] = harness.citta.records
    assert record.experience.action["tool"] == "quote_price" and record.experience.reward["harm"] == 1.0
