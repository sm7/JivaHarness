from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .citta import CittaLedger, Experience


@dataclass(frozen=True)
class DeploymentSpec:
    agent_name: str
    role: str
    purpose: str
    universal_obligations: tuple[str, ...]
    capabilities: tuple[str, ...]
    conditions: dict[str, Any]
    guna: dict[str, float]


@dataclass(frozen=True)
class PrincipalRequest:
    principal: str
    channel: str
    goal: str
    observation: dict[str, Any]
    requested_capability: str


@dataclass(frozen=True)
class VerifiableCredential:
    agent_did: str
    principal: str
    scope: tuple[str, ...]
    capability: str
    role: str
    conditions: dict[str, Any]
    ephemeral: bool = True
    transferable: bool = False
    identity_scoped: bool = True


@dataclass(frozen=True)
class Reward:
    task: float
    sreya: float
    sattva_divergence: float
    sankalpa_shakti: float = 0.5
    sattva_weight: float = 0.25

    @property
    def total(self) -> float:
        return self.task + self.sankalpa_shakti * self.sreya - self.sattva_weight * self.sattva_divergence

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class RunResult:
    authorized: bool
    agent_did: str
    credential: VerifiableCredential | None
    action: dict[str, Any] | None
    reward: Reward | None
    citta_root: str
    reason: str | None = None


@dataclass(frozen=True)
class Atman:
    """Universal foundational potential, outside the runtime governance boundary."""

    base_foundation_model: str = "Base LLM provider"
    governed_at_runtime: bool = False


@dataclass(frozen=True)
class Prakriti:
    """Controlled execution substrate; constrains execution but not authorization."""

    secure_runtime_substrate: str = "Compute & Runtime"
    determines_authorization: bool = False


@dataclass(frozen=True)
class Dharma:
    """Universal obligations and context-specific authorization; does not execute."""

    sadharana: tuple[str, ...]
    visesha_purpose: str
    participates_in_execution: bool = False


@dataclass(frozen=True)
class Karma:
    """Consolidated accountability and feedback substrate."""

    intervenes_at_runtime: bool = False


@dataclass(frozen=True)
class Karmadhyaksa:
    """Practical overseer role assigned by the paper to the deployer or organization."""

    agent_deployer: str = "agent deployer"


class ScopeResolver:
    """Atman-Jiva boundary and Citta-anchored OBO token issuance."""

    def issue(self, spec: DeploymentSpec, request: PrincipalRequest, did: str) -> VerifiableCredential | None:
        if request.requested_capability not in spec.capabilities:
            return None
        amount = request.observation.get("amount")
        maximum = spec.conditions.get("max_amount")
        if amount is not None and maximum is not None and amount > maximum:
            return None
        return VerifiableCredential(
            agent_did=did,
            principal=request.principal,
            scope=(spec.purpose,),
            capability=request.requested_capability,
            role=spec.role,
            conditions=dict(spec.conditions),
        )


class Sankalpa:
    """Policy-constrained decision logic with a foundation-model-backbone boundary."""

    def act(
        self, request: PrincipalRequest, credential: VerifiableCredential, constraints: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action = {
            "name": credential.capability,
            "goal": request.goal,
            "content": f"{credential.role} action for: {request.goal}",
        }
        trace_material = {
            "state": request.observation,
            "goal": request.goal,
            "constraints": constraints,
            "action": action,
        }
        digest = hashlib.sha256(
            json.dumps(trace_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return action, {"digest": digest}


class Karmaphaladata:
    """Practical approximate intrinsic critic with the paper's tripartite reward."""

    def evaluate(
        self, spec: DeploymentSpec, request: PrincipalRequest, action: dict[str, Any], activation_trace: dict[str, Any]
    ) -> Reward:
        task = 1.0 if action["name"] == request.requested_capability else 0.0
        sreya = 1.0 if all(spec.universal_obligations) and spec.purpose in action["content"] else 0.5
        sattva_divergence = abs(1.0 - spec.guna["sattva"])
        return Reward(task=task, sreya=sreya, sattva_divergence=sattva_divergence)


class JivaHarness:
    """Conditioned deployed agent: scoped actor, intrinsic critic, and persistent Citta."""

    def __init__(self, spec: DeploymentSpec, citta: CittaLedger):
        self.spec = spec
        self.citta = citta
        self.atman = Atman()
        self.prakriti = Prakriti()
        self.dharma = Dharma(spec.universal_obligations, spec.purpose)
        self.karma = Karma()
        self.karmadhyaksa = Karmadhyaksa()
        self.scope_resolver = ScopeResolver()
        self.sankalpa = Sankalpa()
        self.karmaphaladata = Karmaphaladata()

    @classmethod
    def deploy(
        cls,
        spec: DeploymentSpec,
        citta_path: str | Path,
        brahmacarya_buffer: list[dict[str, Any]],
    ) -> "JivaHarness":
        if set(spec.guna) != {"sattva", "rajas", "tamas"}:
            raise ValueError("guna must contain exactly sattva, rajas, and tamas")
        if abs(sum(spec.guna.values()) - 1.0) > 1e-9:
            raise ValueError("guna distribution must sum to 1")
        return cls(spec, CittaLedger.create(citta_path, brahmacarya_buffer))

    def run(self, request: PrincipalRequest) -> RunResult:
        credential = self.scope_resolver.issue(self.spec, request, self.citta.did)
        if credential is None:
            return RunResult(
                authorized=False, agent_did=self.citta.did, credential=None,
                action=None, reward=None, citta_root=self.citta.root,
                reason="requested capability or conditions are outside visesha-dharma scope",
            )

        constraints = {
            "allowed_actions": list(self.spec.capabilities),
            "conditions": self.spec.conditions,
            "guna": self.spec.guna,
            "universal_obligations": list(self.spec.universal_obligations),
            "purpose": self.spec.purpose,
        }
        action, activation_trace = self.sankalpa.act(request, credential, constraints)
        reward = self.karmaphaladata.evaluate(self.spec, request, action, activation_trace)
        self.citta.append(Experience(
            state={"partial_observation": request.observation, "principal": request.principal, "channel": request.channel},
            goal=request.goal,
            constraints=constraints,
            action=action,
            reward=reward.to_dict(),
            activation_trace=activation_trace,
        ))
        return RunResult(
            authorized=True, agent_did=self.citta.did, credential=credential,
            action=action, reward=reward, citta_root=self.citta.root,
        )
