from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Layer:
    key: str
    name: str
    components: tuple[str, ...]
    governed_at_runtime: bool | None = None
    intervenes_at_runtime: bool | None = None


@dataclass(frozen=True)
class Architecture:
    source: str
    flows: tuple[str, ...]
    layers: tuple[Layer, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "Architecture":
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        layers = tuple(
            Layer(
                key=item["key"],
                name=item["name"],
                components=tuple(item["components"]),
                governed_at_runtime=item.get("governed_at_runtime"),
                intervenes_at_runtime=item.get("intervenes_at_runtime"),
            )
            for item in data["layers"]
        )
        return cls(data["source"], tuple(data["flows"]), layers)

    @property
    def component_names(self) -> set[str]:
        return {layer.name for layer in self.layers} | {
            component for layer in self.layers for component in layer.components
        }

    def layer(self, key: str) -> Layer:
        return next(layer for layer in self.layers if layer.key == key)

    @property
    def atman(self) -> Layer:
        return self.layer("atman")

    @property
    def karma(self) -> Layer:
        return self.layer("karma")
