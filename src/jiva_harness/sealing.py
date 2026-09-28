from __future__ import annotations

from typing import Any, Protocol


class Sealer(Protocol):
    """Boundary for sensitive Citta payloads.

    Citta hashes the sealed envelope, not the plaintext, so Merkle proofs verify without the key
    and swapping in an encrypting sealer changes no ledger code.
    """

    scheme: str

    def seal(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def unseal(self, envelope: dict[str, Any]) -> dict[str, Any]: ...


class PlaintextSealer:
    """Default until encryption lands: same envelope shape, payload stored in the clear."""

    scheme = "plaintext"

    def seal(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"scheme": self.scheme, "payload": dict(payload)}

    def unseal(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if envelope.get("scheme") != self.scheme:
            raise ValueError(f"cannot unseal scheme {envelope.get('scheme')!r} with {self.scheme!r}")
        return dict(envelope["payload"])
