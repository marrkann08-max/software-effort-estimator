"""Atomic single-source-of-truth storage for reported model metrics."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "metrics.json"


def load_metrics(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        return {"schema_version": 1, "models": {}}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def update_metrics(model_key: str, values: dict, methodology: dict | None = None) -> dict:
    payload = load_metrics()
    payload["schema_version"] = 1
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if methodology:
        payload["methodology"] = methodology
    payload.setdefault("models", {})[model_key] = values

    temporary = METRICS_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, METRICS_PATH)
    return payload
