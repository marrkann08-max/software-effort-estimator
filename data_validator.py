"""Schema mapping and validation for the local Your Data workflow."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "project_id",
    "completed_date",
    "actual_effort_hours",
    "size_kloc",
    "team_size",
    "duration_days",
)

COST_DRIVERS = (
    "RELY", "DATA", "CPLX", "TIME", "STOR", "VIRT", "TURN",
    "ACAP", "AEXP", "PCAP", "VEXP", "LEXP", "MODP", "TOOL", "SCED",
)

OPTIONAL_COLUMNS = (
    "development_mode",
    *COST_DRIVERS,
    "story_points",
    "commit_count",
    "pr_count",
    "median_review_hours",
    "merge_rate",
    "test_file_ratio",
    "language_count",
    "ai_tool_count",
)

RATINGS = {"very low", "low", "nominal", "high", "very high"}
MODES = {"organic", "semi-detached", "semidetached", "embedded"}

ALIASES = {
    "project_id": ("project", "project name", "project key", "id", "key"),
    "completed_date": ("completion date", "completed at", "end date", "finish date"),
    "actual_effort_hours": ("actual hours", "effort hours", "person hours", "jira hours", "worklog hours"),
    "size_kloc": ("kloc", "size kloc", "thousand lines of code", "loc thousands"),
    "team_size": ("contributors", "developer count", "developers", "headcount"),
    "duration_days": ("calendar days", "elapsed days", "project duration", "lead time days"),
    "development_mode": ("dev mode", "mode", "project mode"),
    "RELY": ("reliability", "required reliability"),
    "DATA": ("database size", "data complexity"),
    "CPLX": ("complexity", "process complexity"),
    "TIME": ("time constraint", "cpu time constraint", "latency constraint"),
    "STOR": ("storage constraint", "memory constraint"),
    "VIRT": ("platform volatility", "vm volatility"),
    "TURN": ("turnaround time", "development turnaround"),
    "ACAP": ("analyst capability", "architect capability"),
    "AEXP": ("application experience", "domain experience"),
    "PCAP": ("programmer capability", "developer capability"),
    "VEXP": ("platform experience", "vm experience"),
    "LEXP": ("language experience", "framework experience"),
    "MODP": ("modern practices", "engineering practices"),
    "TOOL": ("tooling", "software tools", "tool quality"),
    "SCED": ("schedule constraint", "schedule pressure"),
    "story_points": ("story points", "delivered story points"),
    "commit_count": ("commits", "total commits"),
    "pr_count": ("pull requests", "prs", "pull request count"),
    "median_review_hours": ("review hours", "pr review hours", "median pr review hours"),
    "merge_rate": ("pr merge rate", "pull request merge rate"),
    "test_file_ratio": ("test ratio", "test coverage proxy"),
    "language_count": ("languages", "number of languages"),
    "ai_tool_count": ("ai tools", "ai tool adoption"),
}

NUMERIC_COLUMNS = (
    "actual_effort_hours", "size_kloc", "team_size", "duration_days",
    "story_points", "commit_count", "pr_count", "median_review_hours",
    "merge_rate", "test_file_ratio", "language_count", "ai_tool_count",
)


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


LOOKUP = {}
for canonical in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS):
    for alias in (canonical, *ALIASES.get(canonical, ())):
        LOOKUP[_key(alias)] = canonical


@dataclass
class ValidationResult:
    data: pd.DataFrame
    mapping: dict[str, str]
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def map_columns(columns) -> tuple[dict[str, str], list[str]]:
    mapping = {str(column): LOOKUP.get(_key(column), str(column).strip()) for column in columns}
    mapped = list(mapping.values())
    duplicates = sorted({name for name in mapped if mapped.count(name) > 1})
    errors = [f"Multiple uploaded columns map to '{name}'. Keep only one." for name in duplicates]
    return mapping, errors


def validate_project_data(frame: pd.DataFrame) -> ValidationResult:
    mapping, errors = map_columns(frame.columns)
    data = frame.rename(columns=mapping).copy()
    warnings: list[str] = []

    # Duplicate aliases create duplicate canonical columns, so stop before any
    # single-column validation that assumes a Series.
    if errors:
        return ValidationResult(data, mapping, errors, warnings)

    missing = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return ValidationResult(data, mapping, errors, warnings)

    if data.empty:
        errors.append("The uploaded file has no project rows.")
        return ValidationResult(data, mapping, errors, warnings)

    identifiers = data["project_id"].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        errors.append("project_id must be present on every row.")
    duplicates = identifiers[identifiers.duplicated(keep=False)].dropna().unique().tolist()
    if duplicates:
        errors.append("project_id must be unique; duplicates include: " + ", ".join(map(str, duplicates[:5])))

    parsed_dates = pd.to_datetime(data["completed_date"], format="mixed", errors="coerce")
    if parsed_dates.isna().any():
        errors.append("completed_date contains invalid or blank dates; use YYYY-MM-DD.")
    else:
        data["completed_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        if (parsed_dates > pd.Timestamp.now()).any():
            warnings.append("Some completion dates are in the future; confirm that these are completed projects.")

    present_numeric = [name for name in NUMERIC_COLUMNS if name in data.columns]
    for name in present_numeric:
        converted = pd.to_numeric(data[name], errors="coerce")
        if converted.isna().any():
            errors.append(f"{name} must contain numbers on every populated row.")
            continue
        data[name] = converted

    for name in ("actual_effort_hours", "size_kloc", "team_size", "duration_days"):
        if name in data and pd.api.types.is_numeric_dtype(data[name]) and (data[name] <= 0).any():
            errors.append(f"{name} must be greater than zero.")
    for name in ("team_size", "commit_count", "pr_count", "language_count", "ai_tool_count"):
        if name in data and pd.api.types.is_numeric_dtype(data[name]):
            if not np.allclose(data[name], np.round(data[name])):
                errors.append(f"{name} must use whole numbers.")
    for name in ("merge_rate", "test_file_ratio"):
        if name in data and pd.api.types.is_numeric_dtype(data[name]):
            if ((data[name] < 0) | (data[name] > 1)).any():
                errors.append(f"{name} must be between 0 and 1.")

    if "development_mode" in data.columns:
        values = data["development_mode"].astype("string").str.strip().str.lower()
        invalid = sorted(set(values.dropna()) - MODES)
        if invalid:
            errors.append("development_mode must be Organic, Semi-detached, or Embedded.")

    for driver in COST_DRIVERS:
        if driver not in data.columns:
            continue
        values = data[driver].astype("string").str.strip().str.lower()
        invalid = sorted(set(values.dropna()) - RATINGS)
        if invalid:
            errors.append(f"{driver} must use Very Low, Low, Nominal, High, or Very High.")

    available_drivers = sum(driver in data.columns for driver in COST_DRIVERS)
    if available_drivers < len(COST_DRIVERS):
        warnings.append(
            f"Mapped {available_drivers} of {len(COST_DRIVERS)} COCOMO cost drivers. "
            "Missing drivers can be collected later; they are not silently imputed here."
        )

    predictor_names = {
        "size_kloc", "team_size", "duration_days", *COST_DRIVERS, *OPTIONAL_COLUMNS,
    }
    candidate_predictors = sum(name in data.columns for name in predictor_names)
    suggested_minimum = max(30, 10 * candidate_predictors)
    if len(data) < suggested_minimum:
        warnings.append(
            f"Only {len(data)} completed projects were supplied. A planning heuristic is at least "
            f"max(30, 10 × candidate predictors) = {suggested_minimum} rows for this {candidate_predictors}-predictor schema. "
            "This is not a guarantee of statistical validity."
        )

    return ValidationResult(data, mapping, errors, warnings)


def template_dataframe() -> pd.DataFrame:
    row = {
        "project_id": "PROJ-001",
        "completed_date": "2026-01-31",
        "actual_effort_hours": 2400,
        "size_kloc": 35,
        "team_size": 6,
        "duration_days": 120,
        "development_mode": "Organic",
        **{driver: "Nominal" for driver in COST_DRIVERS},
        "story_points": 420,
        "commit_count": 850,
        "pr_count": 120,
        "median_review_hours": 18,
        "merge_rate": 0.82,
        "test_file_ratio": 0.24,
        "language_count": 3,
        "ai_tool_count": 1,
    }
    return pd.DataFrame([row])
