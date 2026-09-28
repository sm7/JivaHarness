from __future__ import annotations

from typing import Any

from .tools import Tool

CATALOG = {"laptop": 900, "monitor": 250, "server": 5000}


def _quote_price(args: dict[str, Any]) -> dict[str, Any]:
    item = args["item"]
    if item not in CATALOG:
        raise KeyError(f"no catalog entry for {item!r}")
    return {"item": item, "amount": CATALOG[item]}


def _place_order(args: dict[str, Any]) -> dict[str, Any]:
    return {"ordered": args["item"], "amount": args["amount"], "status": "submitted"}


quote_price = Tool(
    name="quote_price", capability="read_catalog",
    description="Look up the catalog price of an item. Read-only.",
    handler=_quote_price,
)

place_order = Tool(
    name="place_order", capability="purchase",
    description="Submit a purchase order. Spends money; cannot be undone by the agent.",
    handler=_place_order, risk="external_effect",
)
