import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from jiva_harness.citta import CittaLedger
from jiva_harness.claude_policy import ClaudePolicy
from jiva_harness.demo import BRAHMACARYA
from jiva_harness.session import Session

from test_claude_policy import FakeClient, reply, tool_use


def text(t):
    return reply(NS(type="text", text=t), stop_reason="end_turn")


def session(tmp_path, responses, inputs):
    client = FakeClient(responses)
    lines, out = iter(inputs), []

    def read(prompt):
        out.append(prompt)
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    s = Session(ClaudePolicy(client=client, keep_history=True), tmp_path, read=read, write=out.append)
    return s, client, out


def test_principal_is_asked_inline_and_a_no_is_a_recorded_block(tmp_path):
    s, client, out = session(tmp_path, [
        reply(tool_use("a", "place_order", item="laptop", amount=900)), text("not ordered"),
    ], ["buy a laptop", "n"])

    s.repl()

    shown = "\n".join(out)
    assert "-> place_order(amount=900, item=\"laptop\")" in shown
    assert "approve? [y]es once" in shown
    assert "BLOCKED  by principle hook" in shown and "jiva> not ordered" in shown
    assert [r.experience.action["outcome"] for r in s.harness.citta.records] == ["blocked"]


def test_yes_runs_the_action_and_the_approval_reason_is_audited(tmp_path):
    s, _, out = session(tmp_path, [
        reply(tool_use("a", "place_order", item="laptop", amount=900)), text("ordered"),
    ], ["buy a laptop", "y"])

    s.repl()

    assert "ALLOWED" in "\n".join(out)
    checks = [e for e in s.audit.entries() if e["event"] == "principle_check"]
    assert checks[0]["data"]["verdicts"][0]["reason"].endswith("approved by principal when proposed")
    assert s.harness.citta.records[0].experience.action["name"] == "purchase"


def test_always_approves_for_the_rest_of_the_session_only(tmp_path):
    s, _, out = session(tmp_path, [
        reply(tool_use("a", "place_order", item="laptop", amount=900)), text("one"),
        reply(tool_use("b", "place_order", item="monitor", amount=250)), text("two"),
    ], ["buy a laptop", "a", "buy a monitor"])

    s.repl()

    assert sum("approve? [y]es" in line for line in out) == 1
    reasons = [e["data"]["verdicts"][0]["reason"] for e in s.audit.entries() if e["event"] == "principle_check"]
    assert reasons == ["external_effect action approved by principal when proposed",
                       "external_effect action approved by principal"]
    assert [r.experience.action["name"] for r in s.harness.citta.records] == ["purchase", "purchase"]


def test_one_conversation_across_tasks_and_new_resets_it(tmp_path):
    s, client, _ = session(tmp_path, [
        reply(tool_use("a", "quote_price", item="laptop")), text("$900"), text("yes, the laptop"), text("fresh"),
    ], ["price of a laptop?", "which item did I ask about?", "/new", "hello"])

    s.repl()

    second = client.requests[2]["messages"]
    assert second[0] == {"role": "user", "content": "price of a laptop?"}
    assert second[-1] == {"role": "user", "content": [{"type": "text", "text": "which item did I ask about?"}]}
    assert client.requests[3]["messages"] == [{"role": "user", "content": "hello"}]


def test_identity_and_ledger_persist_between_sessions(tmp_path):
    first, _, _ = session(tmp_path, [reply(tool_use("a", "quote_price", item="laptop")), text("ok")], ["q"])
    first.repl()
    second, _, out = session(tmp_path, [reply(tool_use("b", "quote_price", item="monitor")), text("ok")], ["q"])
    second.repl()

    assert second.harness.citta.did == first.harness.citta.did
    assert len(second.harness.citta.records) == 2 and second.harness.citta.verify()
    assert second.audit.verify() and "(1 records so far)" in out[0]


def test_tampered_ledger_is_not_extended(tmp_path):
    first, _, _ = session(tmp_path, [reply(tool_use("a", "quote_price", item="laptop")), text("ok")], ["q"])
    first.repl()
    path = tmp_path / "citta.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["experience"]["goal"] = "rewritten"
    path.write_text("\n".join([lines[0], json.dumps(row)]) + "\n")

    with pytest.raises(ValueError, match="failed verification"):
        CittaLedger.open(path, BRAHMACARYA)


def test_commands(tmp_path):
    s, _, out = session(tmp_path, [], ["/tools", "/approve place_order", "/ledger", "/bogus", "/quit", "never read"])

    s.repl()

    shown = "\n".join(out)
    assert "save_draft_po" in shown and "NOT in this deployment's scope" in shown
    assert "pre-approved: place_order" in shown and s.always == {"place_order"}
    assert "hash chain verifies: True" in shown and "unknown command '/bogus'" in shown
    assert "never read" not in shown


def test_api_failure_is_reported_and_the_session_continues(tmp_path):
    class Boom(Exception):
        pass

    client = NS(beta=NS(messages=NS(create=lambda **_: (_ for _ in ()).throw(Boom("no key")))))
    out, lines = [], iter(["task one", "/quit"])
    s = Session(ClaudePolicy(client=client, keep_history=True), tmp_path, read=lambda _: next(lines),
                write=out.append, recoverable=lambda exc: f"failed: {exc}" if isinstance(exc, Boom) else None)

    s.repl()

    assert "  failed: no key" in out


def test_bare_cli_starts_the_session(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "jiva_harness.cli", "--state-dir", str(tmp_path / "s")],
        input="/tools\n/quit\n", capture_output=True, text=True, env={"PATH": "", "ANTHROPIC_API_KEY_JIVA": "k"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "Jiva harness: procurement-jiva" in completed.stdout and "place_order" in completed.stdout
