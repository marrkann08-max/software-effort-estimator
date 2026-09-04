import pandas as pd

from data_validator import COST_DRIVERS, REQUIRED_COLUMNS, template_dataframe, validate_project_data


def test_template_is_valid_and_contains_all_cocomo_drivers():
    template = template_dataframe()
    result = validate_project_data(template)
    assert result.valid
    assert set(REQUIRED_COLUMNS).issubset(template.columns)
    assert set(COST_DRIVERS).issubset(template.columns)


def test_common_headers_are_mapped_automatically():
    frame = template_dataframe().rename(columns={
        "project_id": "Project Key",
        "completed_date": "Completion Date",
        "actual_effort_hours": "Jira Hours",
        "size_kloc": "KLOC",
        "team_size": "Contributors",
        "duration_days": "Calendar Days",
        "PCAP": "Programmer Capability",
    })
    result = validate_project_data(frame)
    assert result.valid
    assert result.mapping["Jira Hours"] == "actual_effort_hours"
    assert result.mapping["Programmer Capability"] == "PCAP"


def test_missing_and_invalid_values_are_rejected():
    frame = pd.DataFrame({
        "project_id": ["A", "A"],
        "completed_date": ["not-a-date", "2026-01-01"],
        "actual_effort_hours": [100, -2],
        "size_kloc": [10, 12],
        "team_size": [2, 2.5],
        "duration_days": [20, 30],
        "PCAP": ["Nominal", "Unknown"],
    })
    result = validate_project_data(frame)
    assert not result.valid
    combined = " ".join(result.errors)
    assert "unique" in combined
    assert "invalid" in combined
    assert "greater than zero" in combined
    assert "whole numbers" in combined
    assert "PCAP" in combined


def test_duplicate_aliases_are_rejected():
    frame = template_dataframe()
    frame["Jira Hours"] = frame["actual_effort_hours"]
    result = validate_project_data(frame)
    assert not result.valid
    assert any("Multiple uploaded columns" in message for message in result.errors)
