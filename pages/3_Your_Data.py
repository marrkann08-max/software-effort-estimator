from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from data_validator import COST_DRIVERS, REQUIRED_COLUMNS, template_dataframe, validate_project_data


st.markdown("""
<style>
.main { background: #F3F3F3; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }
.page-header { margin-bottom: 1.5rem; padding-bottom: 1.1rem; border-bottom: 1px solid #E2E2E2; }
.page-title { font-size: 1.2rem; font-weight: 650; color: #111; }
.page-subtitle { font-size: .82rem; color: #666; margin-top: .25rem; }
.section-header { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 700; color: #555; margin: 1.7rem 0 .7rem; }
.schema-card { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 1rem 1.2rem; height: 100%; }
.schema-card h4 { margin: 0 0 .5rem; font-size: .9rem; }
.schema-card p, .schema-card li { color: #555; font-size: .82rem; line-height: 1.55; }
code { font-size: .78rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <div class="page-title">Your Data</div>
  <div class="page-subtitle">Validate completed-project records locally before any custom-model work.</div>
</div>
""", unsafe_allow_html=True)

st.info(
    "This page validates structure and data quality only. It does not train a model or promise an MMRE. "
    "Custom training should begin only after the effort target, dates, sample size, and intended predictors are reviewed."
)

template = template_dataframe()
st.download_button(
    "Download CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="software_effort_project_template.csv",
    mime="text/csv",
    width="stretch",
)

st.markdown("<div class='section-header'>Required data</div>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown("""
    <div class="schema-card"><h4>Required columns</h4>
    <ul>
      <li><code>project_id</code> — unique completed-project identifier</li>
      <li><code>completed_date</code> — ISO date, used for future chronological validation</li>
      <li><code>actual_effort_hours</code> — observed person-hours, not planned hours</li>
      <li><code>size_kloc</code> — delivered thousands of lines of code</li>
      <li><code>team_size</code> — contributors assigned to the project</li>
      <li><code>duration_days</code> — elapsed calendar days</li>
    </ul></div>
    """, unsafe_allow_html=True)
with right:
    st.markdown("""
    <div class="schema-card"><h4>Optional predictors</h4>
    <p>Include <code>development_mode</code>, any of the 15 COCOMO ratings, and workflow signals such as
    story points, commits, pull requests, review time, merge rate, test-file ratio, languages, and AI-tool count.</p>
    <p>Column aliases such as <code>Jira hours</code>, <code>completion date</code>, <code>contributors</code>,
    and full driver names such as <code>programmer capability</code> are mapped automatically.</p></div>
    """, unsafe_allow_html=True)

with st.expander("COCOMO rating values and mapping", expanded=False):
    st.write(
        "Each cost-driver column accepts: Very Low, Low, Nominal, High, or Very High. "
        "The validator recognizes both driver codes and common descriptive names."
    )
    st.dataframe(
        pd.DataFrame({
            "Driver": COST_DRIVERS,
            "Purpose": [
                "Required reliability", "Data/database complexity", "Product complexity",
                "Execution-time constraint", "Memory/storage constraint", "Platform volatility",
                "Development turnaround", "Analyst/architect capability", "Application experience",
                "Programmer capability", "Platform experience", "Language experience",
                "Modern practices", "Software tools", "Schedule constraint",
            ],
        }),
        hide_index=True,
        width="stretch",
    )

st.markdown("<div class='section-header'>Upload and validate</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload completed projects", type=["csv", "xlsx"], help="Files are processed in this running app session.")

if uploaded is None:
    st.caption("Upload a CSV or XLSX file. Start with the template if your column names differ.")
else:
    try:
        if uploaded.name.lower().endswith(".csv"):
            frame = pd.read_csv(uploaded)
        else:
            frame = pd.read_excel(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()

    result = validate_project_data(frame)
    mapped = [(source, target) for source, target in result.mapping.items() if source != target]

    if mapped:
        st.write("Automatic column mapping")
        st.dataframe(pd.DataFrame(mapped, columns=["Uploaded column", "Mapped column"]), hide_index=True, width="stretch")

    for message in result.errors:
        st.error(message)
    for message in result.warnings:
        st.warning(message)

    if result.valid:
        st.success(f"Schema validation passed for {len(result.data):,} completed projects.")
        st.dataframe(result.data.head(25), hide_index=True, width="stretch")
        st.download_button(
            "Export validated CSV",
            data=result.data.to_csv(index=False).encode("utf-8"),
            file_name="validated_software_effort_projects.csv",
            mime="text/csv",
            width="stretch",
        )

st.markdown("<div class='section-header'>When training becomes defensible</div>", unsafe_allow_html=True)
st.markdown("""
Before training a custom model:

1. Freeze the outcome definition: completed-project `actual_effort_hours` from a consistent worklog source.
2. Choose the predictors before evaluation and remove fields derived from the target.
3. Use a practical planning heuristic of at least **max(30, 10 × candidate predictors)** completed projects. This is a rule of thumb, not a guarantee; effective sample size also depends on missingness, outliers, and project diversity.
4. Use nested validation for preprocessing, any target-informed feature weighting, and hyperparameter selection. Use a chronological hold-out when completion dates permit.
5. Report MMRE, PRED(25), PRED(50), interval coverage, and uncertainty. Compare weighted and unweighted models separately.

Sample-size source note: the 10× screen is deliberately presented only as a planning heuristic. [Green (1991)](https://doi.org/10.1207/S15327906MBR2603_7) cautions that fixed sample-size ratios trade statistical specificity for simplicity; a real study should use a design-specific power and validation analysis.

The app intentionally does not fit a custom model from a small upload. Doing so would produce a polished score without enough evidence that it will generalize.
""")

st.caption("Required canonical names: " + ", ".join(REQUIRED_COLUMNS))
