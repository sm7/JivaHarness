from jiva_harness.examples import place_order, quote_price
from jiva_harness.importance import ImportanceGate
from jiva_harness.loop import Finish, ToolCall

from test_agent_loop import build, run


class FixedModel:
    def __init__(self, score):
        self.score, self.seen = score, []

    def probability(self, features):
        self.seen.append(features)
        return self.score


def test_routine_reads_stay_out_of_citta_but_are_rolled_up(tmp_path):
    reads = [ToolCall("quote_price", {"item": "laptop"})] * 5
    harness, audit, loop = build(tmp_path, reads + [Finish("done")])
    loop.importance = ImportanceGate()

    result = run(loop)

    assert [t.outcome for t in result.turns] == ["executed"] * 5
    [rollup] = [r.experience.action for r in harness.citta.records]
    assert rollup["name"] == "rollup" and rollup["skipped"] == 5
    entries = audit.entries()
    skipped = [e for e in entries if e["event"] == "tool_executed"]
    assert all(e["data"]["citta_skipped"] for e in skipped)
    assert (rollup["audit_from_seq"], rollup["audit_to_seq"]) == (skipped[0]["seq"], skipped[-1]["seq"])
    assert rollup["audit_to_hash"] == skipped[-1]["hash"]
    assert entries[-1]["event"] == "run_finished" and entries[-1]["data"]["citta_root"] == harness.citta.root
    assert harness.citta.verify() and audit.verify()


def test_consequential_actions_and_refusals_are_always_kept(tmp_path):
    harness, _, loop = build(tmp_path, [
        ToolCall("quote_price", {"item": "laptop"}),
        ToolCall("place_order", {"item": "laptop", "amount": 900}),  # blocked by Ahimsa
        ToolCall("place_order", {"item": "laptop", "amount": 900}),
        Finish("done"),
    ], capabilities=("read_catalog", "purchase"))
    loop.importance = ImportanceGate(model=FixedModel(0.0))

    approvals = iter([False, True])
    loop.hook.principles[0].ask = lambda ctx: next(approvals)
    run(loop)

    actions = [r.experience.action for r in harness.citta.records]
    assert [a.get("outcome", a["name"]) for a in actions] == ["blocked", "purchase", "rollup"]


def test_model_can_keep_a_read_and_sees_no_args(tmp_path):
    harness, _, loop = build(tmp_path, [ToolCall("quote_price", {"item": "laptop"}), Finish("done")])
    model = FixedModel(0.9)
    loop.importance = ImportanceGate(model=model)

    run(loop)

    assert [r.experience.action["tool"] for r in harness.citta.records] == ["quote_price"]
    assert set(model.seen[0]) == {"tool", "risk", "capability", "step", "reward_total", "harm"}


def test_low_reward_and_harm_flag_override_the_model():
    gate = ImportanceGate(model=FixedModel(0.0), min_reward=1.0)
    assert gate.decide(quote_price, 1, {"total": 0.4}).keep
    assert gate.decide(quote_price, 1, {"total": 1.4, "harm": True}).keep
    assert not gate.decide(quote_price, 1, {"total": 1.4}).keep
    assert gate.decide(place_order, 1, {"total": 1.4}).keep


def test_without_a_gate_every_action_is_a_leaf(tmp_path):
    harness, _, loop = build(tmp_path, [ToolCall("quote_price", {"item": "laptop"})] * 3 + [Finish("done")])
    run(loop)
    assert len(harness.citta.records) == 3
