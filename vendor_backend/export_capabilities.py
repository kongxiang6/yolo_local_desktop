from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPORT_CAPABILITIES_PATH = Path(__file__).with_name("export_capabilities.json")


def _normalize_entry(raw_entry: Any) -> dict[str, Any]:
    if not isinstance(raw_entry, dict):
        raise ValueError("Each export capability entry must be an object.")

    format_id = str(raw_entry.get("id") or "").strip()
    label = str(raw_entry.get("label") or "").strip()
    raw_arguments = raw_entry.get("arguments", [])
    if not format_id:
        raise ValueError("Export capability entry is missing a non-empty 'id'.")
    if not label:
        raise ValueError(f"Export capability '{format_id}' is missing a non-empty 'label'.")
    if not isinstance(raw_arguments, list) or any(not str(argument).strip() for argument in raw_arguments):
        raise ValueError(f"Export capability '{format_id}' has invalid 'arguments'.")

    arguments = tuple(dict.fromkeys(str(argument).strip() for argument in raw_arguments))
    return {"id": format_id, "label": label, "arguments": arguments}


@lru_cache(maxsize=1)
def load_export_capabilities() -> tuple[dict[str, Any], ...]:
    payload = json.loads(EXPORT_CAPABILITIES_PATH.read_text(encoding="utf-8"))
    raw_formats = payload.get("formats")
    if not isinstance(raw_formats, list) or not raw_formats:
        raise ValueError("Export capability contract must contain a non-empty 'formats' array.")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw_entry in raw_formats:
        entry = _normalize_entry(raw_entry)
        format_id = entry["id"]
        if format_id in seen:
            raise ValueError(f"Duplicate export capability id: {format_id}")
        seen.add(format_id)
        normalized.append(entry)
    return tuple(normalized)


@lru_cache(maxsize=1)
def export_capability_map() -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in load_export_capabilities()}


def export_format_choices() -> tuple[str, ...]:
    return tuple(entry["id"] for entry in load_export_capabilities())


def export_format_labels() -> dict[str, str]:
    return {entry["id"]: entry["label"] for entry in load_export_capabilities()}


def supported_export_arguments(export_format: str) -> set[str]:
    entry = export_capability_map().get(export_format)
    return set(entry["arguments"]) if entry else set()
