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


def _save_draft_po(args: dict[str, Any]) -> dict[str, Any]:
    return {"draft_po": args["item"], "amount": args["amount"], "status": "draft"}


def _schema(**properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_ITEM = {"type": "string", "description": "Catalog item name, e.g. laptop, monitor, server."}
_AMOUNT = {"type": "number", "description": "Price in USD, taken from quote_price."}

quote_price = Tool(
    name="quote_price", capability="read_catalog",
    description="Look up the catalog price of an item. Read-only.",
    handler=_quote_price, input_schema=_schema(item=_ITEM),
)

place_order = Tool(
    name="place_order", capability="purchase",
    description="Submit a purchase order. Spends money; cannot be undone by the agent.",
    handler=_place_order, risk="external_effect", input_schema=_schema(item=_ITEM, amount=_AMOUNT),
)

save_draft_po = Tool(
    name="save_draft_po", capability="draft_po",
    description="Save a draft purchase order for later review. Can be edited or discarded.",
    handler=_save_draft_po, risk="reversible_write", input_schema=_schema(item=_ITEM, amount=_AMOUNT),
)
