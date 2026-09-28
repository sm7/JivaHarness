import json
import subprocess
import sys
from pathlib import Path

import pytest

from jiva_harness.architecture import Architecture
from jiva_harness.citta import CittaLedger, Experience
from jiva_harness.runtime import JivaHarness, DeploymentSpec, PrincipalRequest


ROOT = Path(__file__).parents[1]


def test_architecture_inventory_covers_every_figure_2_component():
    architecture = Architecture.from_json(ROOT / "paper_architecture.json")

    expected = {
        "Atman - Universal Foundational Potential",
        "Principal Hierarchy", "Human Principal", "Orchestrator Agent",
        "Chat Apps", "APIs", "Service Portals", "Browsers", "Agent Marketplace",
        "Jiva - The Conditioned, Deployed Agent", "Identity-centric Control Layer",
        "Sankalpa - Policy function", "Agent Decision Logic", "Goal-directed planning",
        "Policy constrained execution",
        "Isvara (Karmaphaladata) - Evaluation function", "Intrinsic action evaluation",
        "Continuous feedback", "Inline risk scoring",
        "Citta - Identity ledger", "Identity Anchor", "Agent DID", "Genesis record",
        "OBO Token Scope", "Ephemeral", "Non-transferable", "Identity-scoped",
        "Immutable Root identity", "Identity-preserving delegation",
        "Scope resolver", "Atman-Jiva boundary",
        "Service principal to Jiva identity migration", "Token issuance anchored to Citta",
        "Agent Instance", "Operational actor - Jiva", "Verifiable credentials",
        "Scope", "Capability", "Role", "Conditions",
        "Prakriti", "Secure Runtime Substrate", "Base LLM provider", "Compute & Runtime",
        "Guna distribution monitoring", "Agent Memory", "Search & Knowledge (Viveka)",
        "Security Guardrails", "Prompt injection", "Insecure output", "Tool misuse",
        "Training leakage",
        "Dharma", "Policy Governance Layer", "Sadharana-dharma", "Universal Safety",
        "EU AI Act", "NIST RMF", "Enterprise baseline controls", "Visesha-dharma",
        "Context-specific agent purpose, scope, and role", "PBAC", "OPA",
        "Contextual authorization engine", "Karmadhyaksa console",
        "Agent deployer", "GRC", "oversight", "Human governance oversight",
        "Agent & LLM gateway", "MCP gateway", "Contract-policy engine",
        "Policy enforcement point", "Regulatory evidence store", "Factsheets",
        "model cards", "audit reports", "Compliance evidence system",
        "Karma", "Identity-bound action record", "Token lineage", "Immutable audit trail",
        "Risk signals", "Behavior assessment", "Outcome feedback", "Audit evidence",
        "Harm attribution", "Access review", "Learning feedback", "Policy feedback",
    }
    assert architecture.component_names == expected
    assert architecture.atman.governed_at_runtime is False
    assert architecture.karma.intervenes_at_runtime is False


def test_citta_identity_is_genesis_hash_and_does_not_change_when_experiences_append(tmp_path):
    ledger = CittaLedger.create(tmp_path / "citta.jsonl", [{"student": "episode-1"}])
    did = ledger.did
    genesis = ledger.genesis_hash

    leaf = ledger.append(Experience(
        state={"observation": "request"},
        goal="summarize",
        constraints={"allowed_actions": ["respond"]},
        action={"name": "respond", "content": "summary"},
        reward={"task": 1.0, "sreya": 0.5, "sattva_divergence": 0.1, "total": 1.4},
        activation_trace={"digest": "trace-1"},
    ))

    assert ledger.did == did
    assert ledger.genesis_hash == genesis
    assert ledger.verify()
    assert ledger.prove(leaf.index).verify()


def test_harness_executes_scoped_actor_critic_step_and_records_complete_citta_tuple(tmp_path):
    spec = DeploymentSpec(
        agent_name="procurement-jiva",
        role="procurement",
        purpose="prepare purchase recommendation",
        universal_obligations=("helpful", "harmless", "honest"),
        capabilities=("recommend",),
        conditions={"max_amount": 1000},
        guna={"sattva": 0.7, "rajas": 0.2, "tamas": 0.1},
    )
    harness = JivaHarness.deploy(
        spec=spec,
        citta_path=tmp_path / "citta.jsonl",
        brahmacarya_buffer=[{"scenario": "approved procurement"}],
    )

    result = harness.run(PrincipalRequest(
        principal="human:alice",
        channel="Chat Apps",
        goal="recommend a laptop",
        observation={"amount": 900},
        requested_capability="recommend",
    ))

    assert result.authorized is True
    assert result.agent_did == harness.citta.did
    assert result.credential.ephemeral is True
    assert result.credential.transferable is False
    assert result.credential.identity_scoped is True
    assert result.reward.total == pytest.approx(
        result.reward.task
        + result.reward.sankalpa_shakti * result.reward.sreya
        - result.reward.sattva_weight * result.reward.sattva_divergence
    )
    record = harness.citta.records[-1]
    assert set(record.experience.to_dict()) == {
        "state", "goal", "constraints", "action", "reward", "activation_trace"
    }
    assert harness.citta.verify()
    assert harness.atman.governed_at_runtime is False
    assert harness.prakriti.determines_authorization is False
    assert harness.dharma.participates_in_execution is False
    assert harness.karma.intervenes_at_runtime is False
    assert harness.karmadhyaksa.agent_deployer == "agent deployer"


def test_scope_resolver_denies_out_of_scope_action_without_actor_execution(tmp_path):
    harness = JivaHarness.deploy(
        spec=DeploymentSpec(
            agent_name="reader-jiva", role="reader", purpose="read only",
            universal_obligations=("helpful", "harmless", "honest"),
            capabilities=("read",), conditions={},
            guna={"sattva": 0.8, "rajas": 0.1, "tamas": 0.1},
        ),
        citta_path=tmp_path / "citta.jsonl",
        brahmacarya_buffer=[{"scenario": "read-only"}],
    )

    result = harness.run(PrincipalRequest(
        principal="service:unknown", channel="APIs", goal="delete data",
        observation={}, requested_capability="delete",
    ))

    assert result.authorized is False
    assert result.action is None
    assert len(harness.citta.records) == 0


def test_cli_demo_runs_an_end_to_end_deployed_agent_step(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "jiva_harness.cli", "demo", "--state-dir", str(tmp_path)],
        check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["authorized"] is True
    assert output["agent_did"].startswith("did:jiva:")
    assert output["citta_verified"] is True
    assert output["paper_tuple_fields"] == [
        "state", "goal", "constraints", "action", "reward", "activation_trace"
    ]
