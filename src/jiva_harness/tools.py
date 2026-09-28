from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .runtime import DeploymentSpec, PrincipalRequest, ScopeResolver, VerifiableCredential


RiskClass = Literal["read", "reversible_write", "irreversible", "external_effect"]
CONSEQUENTIAL: frozenset[str] = frozenset({"irreversible", "external_effect"})


@dataclass(frozen=True)
class Tool:
    """A Prakriti capability the Jiva may invoke. The handler executes; it never authorizes."""

    name: str
    capability: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    risk: RiskClass = "read"
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name, "capability": self.capability,
            "description": self.description, "risk": self.risk, "input_schema": self.input_schema,
        }


class PermissionDenied(Exception):
    pass


class ToolRegistry:
    """Tool catalog plus the permission gate: every call needs a fresh Citta-anchored credential."""

    def __init__(self, scope_resolver: ScopeResolver | None = None):
        self._tools: dict[str, Tool] = {}
        self.scope_resolver = scope_resolver or ScopeResolver()

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise PermissionDenied(f"unknown tool: {name}") from None

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def gate(
        self, tool: Tool, args: dict[str, Any], spec: DeploymentSpec,
        principal: str, channel: str, goal: str, did: str,
    ) -> VerifiableCredential:
        """Visesha-dharma check. Issues an ephemeral OBO credential scoped to this one call."""
        credential = self.scope_resolver.issue(spec, PrincipalRequest(
            principal=principal, channel=channel, goal=goal,
            observation=args, requested_capability=tool.capability,
        ), did)
        if credential is None:
            raise PermissionDenied(
                f"capability '{tool.capability}' or its conditions are outside the deployment scope"
            )
        return credential
