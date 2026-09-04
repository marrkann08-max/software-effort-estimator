"""Read the versioned validation report used by every user-facing surface."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "metrics.json"


def load_metrics() -> dict:
    """Return metrics.json, failing loudly rather than showing stale fallbacks."""
    with METRICS_PATH.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported metrics.json schema")
    return report


def model_metrics(name: str) -> dict:
    return load_metrics()["models"][name]
