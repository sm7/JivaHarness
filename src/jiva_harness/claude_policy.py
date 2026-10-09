from __future__ import annotations

import json
import os
from typing import Any

from .loop import Finish, Proposal, ToolCall, Turn

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
API_KEY_ENV = ("ANTHROPIC_API_KEY_JIVA", "ANTHROPIC_API_KEY")  # first one set wins


def api_key_from_env() -> str | None:
    return next((os.environ[name] for name in API_KEY_ENV if os.environ.get(name)), None)


def _api_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec["name"],
        "description": f"{spec['description']} Risk class: {spec['risk']}.",
        "input_schema": spec.get("input_schema") or {"type": "object", "properties": {}},
    }


def _tool_result(tool_use_id: str, turn: Turn) -> dict[str, Any]:
    return {
        "type": "tool_result", "tool_use_id": tool_use_id,
        "content": json.dumps({"outcome": turn.outcome, **turn.observation}, sort_keys=True),
        "is_error": turn.outcome != "executed",
    }


class ClaudePolicy:
    """Sankalpa backed by Claude through Anthropic tool use.

    Claude only proposes. Each tool_use block becomes one ToolCall that the AgentLoop gates; the
    loop's verdict (executed, blocked, denied, error, halted) goes back to Claude as the tool_result.
    Parallel tool use is off so every proposal is gated before the next one is made. The transcript
    is kept append-only, including thinking blocks, as the API requires for tool-use continuations.
    """

    def __init__(
        self, client: Any = None, model: str = DEFAULT_MODEL, system: str | None = None,
        effort: str = "medium", max_tokens: int = 16000, fallbacks: bool = True, keep_history: bool = False,
    ):
        if client is None:
            import anthropic  # optional dependency: pip install 'jiva-harness[claude]'

            key = api_key_from_env()
            # with no key set, the SDK resolves other credentials (auth token, ant profile) itself
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.client, self.model, self.system = client, model, system
        self.effort, self.max_tokens, self.fallbacks = effort, max_tokens, fallbacks
        self.keep_history = keep_history  # True: each run's goal continues one conversation
        self.reset()

    def reset(self) -> None:
        """Forget the conversation; the next goal starts a new one."""
        self.messages: list[dict[str, Any]] = []
        self.responses: list[Any] = []
        self._pending: str | None = None
        self._seen = 0

    def propose(self, goal: str, history: list[Turn], tools: list[dict[str, Any]]) -> Proposal:
        if not history:  # a new run
            if not (self.keep_history and self.messages):
                self.reset()
                self.messages.append({"role": "user", "content": goal})
            else:
                content: list[dict[str, Any]] = []
                if self._pending is not None:  # the last run stopped (halt, max_steps) before judging this call
                    content.append({"type": "tool_result", "tool_use_id": self._pending, "is_error": True,
                                    "content": json.dumps({"outcome": "not_run", "error": "run ended first"})})
                    self._pending = None
                self.messages.append({"role": "user", "content": [*content, {"type": "text", "text": goal}]})
            self._seen = 0
        for turn in history[self._seen:]:
            if self._pending is None:
                raise RuntimeError("history has a turn this policy did not propose")
            self.messages.append({"role": "user", "content": [_tool_result(self._pending, turn)]})
            self._pending = None
        self._seen = len(history)

        response = self.client.beta.messages.create(**self._request(tools))
        self.responses.append(response)
        self.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return Finish(f"model refused: {getattr(details, 'explanation', None) or 'no explanation'}")
        if response.stop_reason == "max_tokens":
            return Finish("model stopped: max_tokens reached before a complete proposal")
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is not None:
            self._pending = tool_use.id
            return ToolCall(tool_use.name, dict(tool_use.input))
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Finish(text or f"model stopped: {response.stop_reason}")

    def _request(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "messages": self.messages,
            "tools": [_api_tool(spec) for spec in tools],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort},
        }
        if self.system:
            request["system"] = self.system
        if self.fallbacks:  # on a safety decline the API retries on a fallback model in the same call
            request["betas"], request["fallbacks"] = [FALLBACK_BETA], "default"
        return request

    def usage(self) -> dict[str, int]:
        totals = {"requests": len(self.responses), "input_tokens": 0, "output_tokens": 0}
        for response in self.responses:
            usage = getattr(response, "usage", None)
            totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        return totals
