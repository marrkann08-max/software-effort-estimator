import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os

from app_metrics import model_metrics
from explainability import agreement_metrics
from validation import prediction_interval

# Root directory — pages/ is one level below the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "#DCDCDC",
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         "#F0F0F0",
    "grid.linewidth":     0.7,
    "xtick.color":        "#999999",
    "ytick.color":        "#999999",
    "xtick.major.size":   3,
    "ytick.major.size":   0,
    "text.color":         "#333333",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.frameon":     False,
    "legend.fontsize":    8,
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "cv02","cv03","cv04","cv11";
}
.main { background: #F3F3F3; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F0F0F !important;
    border-right: 1px solid #1E1E1E !important;
}
section[data-testid="stSidebar"] > div { background: #0F0F0F !important; }

/* All text in sidebar */
section[data-testid="stSidebar"] * { color: #CCCCCC; }

/* Labels on sliders and selects */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #BBBBBB !important; font-size: 0.78rem !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Selectbox input */
section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: #1C1C1C !important; border-color: #383838 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #DDDDDD !important;
}

/* Expander headers — target both old and new Streamlit selectors */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] details summary,
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] .streamlit-expanderHeader p {
    color: #CCCCCC !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
section[data-testid="stSidebar"] details summary:hover {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    border-bottom: 1px solid #2A2A2A !important;
    background: transparent !important;
}

/* Slider value text */
section[data-testid="stSidebar"] [data-testid="stSlider"] p,
section[data-testid="stSidebar"] .stSlider p {
    color: #AAAAAA !important; font-size: 0.74rem !important;
}
.sb-section {
    font-size: 0.72rem; color: #888888; letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600; margin: 0.9rem 0 0.3rem 0;
}
.sb-ai-source {
    font-family: "JetBrains Mono", monospace; font-size: 0.66rem; color: #666666;
    margin-bottom: 0.55rem; padding: 0.18rem 0.4rem;
    background: rgba(255,255,255,0.025); border: 1px solid #242424;
    border-radius: 3px; display: inline-block;
}

/* Page header */
.page-header {
    margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid #E2E2E2;
}
.page-title {
    font-size: 1.15rem; font-weight: 650; color: #111111;
    letter-spacing: -0.02em; margin: 0 0 0.2rem 0;
}
.page-subtitle { font-size: 0.79rem; color: #666666; margin: 0 0 0.65rem 0; }
.stat-pill {
    display: inline-flex; align-items: center;
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 4px;
    padding: 0.13rem 0.48rem;
    font-family: "JetBrains Mono", monospace; font-size: 0.66rem; color: #555555;
    margin-right: 0.28rem;
}
.stat-pill-blue {
    display: inline-flex; align-items: center;
    background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 4px;
    padding: 0.13rem 0.52rem;
    font-family: "JetBrains Mono", monospace; font-size: 0.66rem; color: #1D4ED8;
    margin-right: 0.28rem; font-weight: 600;
}
.stat-pill-green {
    display: inline-flex; align-items: center;
    background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 4px;
    padding: 0.13rem 0.52rem;
    font-family: "JetBrains Mono", monospace; font-size: 0.66rem; color: #15803D;
    margin-right: 0.28rem; font-weight: 600;
}
.stat-pill-dark {
    display: inline-flex; align-items: center;
    background: #1C1C1C; border: 1px solid #333333; border-radius: 4px;
    padding: 0.13rem 0.52rem;
    font-family: "JetBrains Mono", monospace; font-size: 0.66rem; color: #AAAAAA;
    margin-right: 0.28rem;
}

/* Metric cards */
.metric-card {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 10px;
    padding: 1.2rem 1.4rem; height: 100%;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}
.metric-card:hover {
    border-color: #C8C8C8; box-shadow: 0 2px 14px rgba(0,0,0,0.07);
}
.metric-label {
    font-size: 0.68rem; color: #666666; letter-spacing: 0.07em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;
}
.metric-value {
    font-family: "JetBrains Mono", monospace; font-size: 2.1rem; font-weight: 600;
    color: #111111; line-height: 1; letter-spacing: -0.03em; white-space: nowrap;
}
.metric-unit { font-size: 0.78rem; color: #888888; margin-top: 0.28rem; }
.metric-range {
    font-family: "JetBrains Mono", monospace; font-size: 0.68rem; color: #888888;
    margin-top: 0.42rem; padding-top: 0.42rem; border-top: 1px solid #F0F0F0;
}
.risk-low  { color: #2E7D32 !important; }
.risk-med  { color: #BF6000 !important; }
.risk-high { color: #C62828 !important; }

/* Exec summary */
.exec-summary {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-left: 2px solid #111111;
    border-radius: 0 6px 6px 0; padding: 0.9rem 1.1rem; margin: 0.75rem 0 1.25rem 0;
}
.exec-summary-label {
    font-size: 0.64rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 600; color: #AAAAAA; margin-bottom: 0.42rem;
}
.exec-summary-text { font-size: 0.875rem; color: #222222; line-height: 1.75; }

/* Section headers */
.section-header {
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 700; color: #555555; margin: 2rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-header::after { content: ""; flex: 1; height: 1px; background: #EBEBEB; }

/* Insight boxes */
.insight-box {
    border-left: 2px solid #DDDDDD; background: #FAFAFA; padding: 0.55rem 0.8rem;
    border-radius: 0 4px 4px 0; margin: 0.28rem 0;
    font-size: 0.83rem; color: #1A1A1A; line-height: 1.55;
}
.insight-up   { border-left-color: #D97070; }
.insight-down { border-left-color: #5FAD6E; }
.insight-note {
    border-left: 2px solid #DDDDDD; background: #FAFAFA; padding: 0.58rem 0.8rem;
    border-radius: 0 4px 4px 0; margin: 0.28rem 0;
    font-size: 0.79rem; color: #555555; line-height: 1.65;
}

/* Pipeline */
.pipeline-box {
    background: #FFFFFF; border: 1px solid #E2E2E2;
    border-radius: 8px; padding: 1.3rem 1.5rem;
}
.pipeline-step { display: flex; align-items: flex-start; margin-bottom: 1.1rem; }
.step-circle {
    display: flex; align-items: center; justify-content: center;
    width: 1.7rem; height: 1.7rem; border-radius: 50%;
    background: #2563EB; color: #FFFFFF;
    font-family: "JetBrains Mono", monospace; font-size: 0.72rem; font-weight: 700;
    min-width: 1.7rem; margin-right: 0.9rem; margin-top: 0.06rem; flex-shrink: 0;
}
.step-title { font-size: 0.84rem; font-weight: 600; color: #2563EB; margin-bottom: 0.18rem; }
.step-desc { font-size: 0.78rem; color: #555555; line-height: 1.65; }
.step-caveat {
    font-size: 0.72rem; color: #999999; line-height: 1.55;
    margin-top: 0.3rem; padding-top: 0.3rem;
    border-top: 1px dashed #EBEBEB; font-style: italic;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #EFEFEF; border-radius: 6px; padding: 2px; width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 5px; padding: 0.3rem 0.85rem; font-size: 0.77rem; font-weight: 500;
    color: #888888; background: transparent; border: none;
    transition: background 140ms ease, color 140ms ease;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #111111 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}

/* Button */
.stButton > button[kind="primary"] {
    background: #DC2626 !important; border: none !important;
    color: #FFFFFF !important; font-weight: 600 !important; font-size: 0.88rem !important;
    border-radius: 6px !important; padding: 0.6rem 1rem !important;
    letter-spacing: 0.02em !important; transition: background 140ms ease !important;
    text-transform: uppercase !important;
}
.stButton > button[kind="primary"]:hover { background: #B91C1C !important; }


/* ── Data tables ─────────────────────────────────────────────────────────── */
.data-table {
    width: 100%; border-collapse: collapse; font-size: 0.8rem;
    font-family: "Inter", sans-serif; margin: 0;
}
.data-table thead tr {
    background: #F7F7F7; border-bottom: 1px solid #E2E2E2;
}
.data-table thead th {
    padding: 0.42rem 0.75rem; text-align: left; font-size: 0.66rem;
    font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
    color: #777777; white-space: nowrap;
}
.data-table tbody tr {
    border-bottom: 1px solid #F0F0F0;
    transition: background 120ms ease;
}
.data-table tbody tr:hover { background: #F7F7F7; }
.data-table tbody td {
    padding: 0.52rem 0.75rem; color: #1A1A1A; vertical-align: middle;
    font-size: 0.82rem;
}
.data-table tbody td:first-child { color: #1A1A1A; font-size: 0.82rem; }

/* Match badges */
.match-badge {
    display: inline-block; padding: 0.12rem 0.5rem; border-radius: 3px;
    font-size: 0.69rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; white-space: nowrap;
}
.match-aligned  { background: #E8F5E9; color: #2E7D32; }
.match-partial  { background: #FFF8E1; color: #BF6000; }
.match-divergent{ background: #FFEBEE; color: #C62828; }

/* Explain note */
.explain-note {
    background: #FFFFFF; border: 1px solid #E2E2E2;
    border-left: 2px solid #AAAAAA; border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.9rem; margin: 0.5rem 0 0.9rem 0;
}
.explain-label {
    font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 600; color: #AAAAAA; margin-bottom: 0.32rem;
}
.explain-text { font-size: 0.8rem; color: #444444; line-height: 1.65; }

/* What-if callout */
.wi-callout {
    background: #F0F7FF; border: 1px solid #C8DCF5; border-radius: 6px;
    padding: 0.65rem 0.9rem; margin-top: 0.6rem;
    font-size: 0.81rem; color: #1A3A5C; line-height: 1.6;
}

/* Footer */
.app-footer {
    margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid #E2E2E2;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.5rem;
}
.footer-left  { font-size: 0.71rem; color: #AAAAAA; line-height: 1.7; }
.footer-right {
    font-family: "JetBrains Mono", monospace; font-size: 0.65rem; color: #CCCCCC;
    background: #F7F7F7; border: 1px solid #E8E8E8; border-radius: 4px;
    padding: 0.18rem 0.5rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


DARK_CSS = """
<style>
.stApp, section.main, .main { background: #111111 !important; }
.main .block-container { background: #111111 !important; }
.page-header { border-bottom-color: #2A2A2A !important; }
.page-title  { color: #EEEEEE !important; }
.page-subtitle { color: #888888 !important; }
.stat-pill      { background: #1E1E1E !important; border-color: #333 !important; color: #AAAAAA !important; }
.stat-pill-blue { background: #0F1E33 !important; border-color: #1E3A5F !important; color: #5B9BD5 !important; }
.stat-pill-green{ background: #0D2218 !important; border-color: #1A4A25 !important; color: #4CAF72 !important; }
.stat-pill-dark { background: #1A1A1A !important; border-color: #2A2A2A !important; color: #AAAAAA !important; }
.metric-card    { background: #1A1A1A !important; border-color: #2A2A2A !important; }
.metric-card:hover { border-color: #3A3A3A !important; box-shadow: 0 2px 14px rgba(0,0,0,0.4) !important; }
.metric-value   { color: #EEEEEE !important; }
.metric-label   { color: #888888 !important; }
.metric-unit    { color: #666666 !important; }
.metric-range   { color: #666666 !important; border-top-color: #2A2A2A !important; }
.risk-low  { color: #4CAF72 !important; }
.risk-med  { color: #D4A04A !important; }
.risk-high { color: #E57373 !important; }
.exec-summary { background: #1A1A1A !important; border-color: #2A2A2A !important; border-left-color: #EEEEEE !important; }
.exec-summary-label { color: #666 !important; }
.exec-summary-text  { color: #CCCCCC !important; }
.section-header { color: #777777 !important; }
.section-header::after { background: #2A2A2A !important; }
.insight-box   { background: #1A1A1A !important; color: #DDDDDD !important; }
.insight-up    { border-left-color: #8B3A3A !important; }
.insight-down  { border-left-color: #2D6A3F !important; }
.insight-note  { background: #1A1A1A !important; color: #BBBBBB !important; }
.data-table thead tr   { background: #1E1E1E !important; border-color: #333 !important; }
.data-table thead th   { color: #777 !important; }
.data-table tbody tr   { border-bottom-color: #222 !important; }
.data-table tbody tr:hover { background: #1E1E1E !important; }
.data-table tbody td   { color: #DDDDDD !important; }
.match-aligned   { background: #0D2218 !important; color: #4CAF72 !important; }
.match-partial   { background: #1E1600 !important; color: #D4A04A !important; }
.match-divergent { background: #1E0D0D !important; color: #E57373 !important; }
.explain-note  { background: #1A1A1A !important; border-color: #2A2A2A !important; }
.explain-label { color: #666 !important; }
.explain-text  { color: #CCCCCC !important; }
.wi-callout    { background: #0F1E33 !important; border-color: #1E3A5F !important; color: #8BB8D8 !important; }
.pipeline-box  { background: #1A1A1A !important; border-color: #2A2A2A !important; }
.step-desc     { color: #AAAAAA !important; }
.step-caveat   { color: #666 !important; }
.app-footer    { border-top-color: #2A2A2A !important; }
.footer-left   { color: #555 !important; }
.footer-right  { background: #1A1A1A !important; border-color: #2A2A2A !important; color: #666 !important; }
/* Streamlit native elements */
.stTabs [data-baseweb="tab-list"] { background: #1E1E1E !important; }
.stTabs [data-baseweb="tab"]      { color: #777 !important; }
.stTabs [aria-selected="true"]    { background: #2A2A2A !important; color: #EEEEEE !important; }
[data-testid="stExpander"]        { background: #1A1A1A !important; border-color: #2A2A2A !important; }
[data-testid="stExpander"] summary p { color: #CCCCCC !important; }
</style>
"""

if st.session_state.get("dark_mode", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────
# Cache key includes file mtime → cache auto-busts whenever model.pkl is retrained
@st.cache_data(show_spinner=False)
def load_model(mtime: float):
    path = os.path.join(ROOT, "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

_model_path = os.path.join(ROOT, "model.pkl")
if not os.path.exists(_model_path):
    st.error(
        "**model.pkl not found.** Run `python train_model.py` to generate it, "
        "then restart the app.",
        icon="🚨",
    )
    st.stop()
_mtime = os.path.getmtime(_model_path)
try:
    art = load_model(_mtime)
except Exception as _e:
    st.error(f"**Failed to load model.pkl:** {_e}\n\nTry re-running `python train_model.py`.", icon="🚨")
    st.stop()
model    = art["model"]
scaler_X = art["scaler_X"]
scaler_y = art["scaler_y"]
fahp_w   = art["fahp_weights"]
kernel_w = art.get("kernel_w", np.sqrt(fahp_w))
X_train  = art["X_train"]
PI_Q10   = art.get("pi_q10", art.get("pi_lo_pct", -0.46))
PI_Q90   = art.get("pi_q90", art.get("pi_hi_pct", 0.56))
COCOMO_METRICS = model_metrics("cocomo81")
GITHUB_METRICS = model_metrics("github")

# Cross-fitted explainability results are part of the model evidence, not an
# optional decoration.  Stop rather than silently falling back to a local,
# in-sample comparison if the training audit has not been generated.
_EXPLAINABILITY_KEYS = {
    "primary_alignment",
    "primary_alignment_across_repeats",
    "primary_alignment_project_bootstrap",
    "primary_comparison",
    "protocol",
    "feature_table",
    "feature_label_permutation_null",
    "joint_stability_scope",
    "nonzero_features_per_explanation",
    "numerical_sensitivity",
    "unweighted_global_ranking_stability",
    "unweighted_local_stability",
    "weighted_alignment_descriptive",
    "weighted_alignment_across_repeats_descriptive",
    "weighting_proxy",
}
try:
    EXPLAINABILITY = COCOMO_METRICS["explainability"]
    _missing_explainability = sorted(_EXPLAINABILITY_KEYS - set(EXPLAINABILITY))
    if _missing_explainability or not EXPLAINABILITY["feature_table"]:
        detail = ", ".join(_missing_explainability) or "feature_table"
        raise KeyError(detail)
except (KeyError, TypeError) as _explainability_error:
    st.error(
        "**Cross-fitted explainability evidence is missing or incomplete.** "
        "Re-run `python train_model.py` to regenerate the model, SHAP audit, and metrics "
        f"before using this page. Missing: {_explainability_error}",
        icon="🚨",
    )
    st.stop()

N_81 = COCOMO_METRICS["n"]
N_93 = COCOMO_METRICS["external_nasa93"]["n"]
PI_Q10 = COCOMO_METRICS["prediction_interval"]["q10"]
PI_Q90 = COCOMO_METRICS["prediction_interval"]["q90"]
_gh_csv_path = os.path.join(ROOT, "github_projects.csv")

FEATURES = [
    "LOC (KLOC)", "RELY", "DATA", "CPLX", "TIME", "STOR",
    "VIRT", "TURN", "ACAP", "AEXP", "PCAP", "VEXP",
    "LEXP", "MODP", "TOOL", "SCED", "Dev Mode",
]
FEAT_DESC = {
    "LOC (KLOC)":  "Codebase size — or total story points / sprint velocity proxy",
    "RELY":        "Consequence of failure — SLA breach, financial loss, safety risk",
    "DATA":        "Data complexity — DB size, ETL pipelines, analytics workloads",
    "CPLX":        "Algorithmic complexity — ML models, real-time systems, distributed logic",
    "TIME":        "Execution-time constraint — latency SLAs, high-throughput API requirements",
    "STOR":        "Memory constraint — edge/embedded targets, serverless, mobile",
    "VIRT":        "Platform volatility — cloud provider churn, k8s/framework version changes",
    "TURN":        "Dev cycle turnaround — CI/CD speed, PR review time, build pipeline latency",
    "ACAP":        "Architect/analyst quality — design decisions, Jira story accuracy, tech debt prevention",
    "AEXP":        "Domain experience — years in the industry vertical (fintech, health, logistics)",
    "PCAP":        "Developer capability — sprint velocity, code review quality, GitHub throughput",
    "VEXP":        "Platform/infra experience — AWS/GCP/Azure, Kubernetes, microservices",
    "LEXP":        "Language & framework experience — Python, TypeScript, Go, React, FastAPI",
    "MODP":        "Engineering practices — Agile maturity, TDD, CI/CD discipline, code reviews",
    "TOOL":        "Dev tooling — Copilot/Cursor adoption, automated testing, deployment pipelines",
    "SCED":        "Schedule pressure — hard launch deadline, regulatory date, contractual milestone",
    "Dev Mode":    "Project type: Organic = small familiar-domain team · Semi-detached = mixed · Embedded = hard real-time / safety-critical",
}
RATING_LABELS = ["Very Low", "Low", "Nominal", "High", "Very High"]
RATING_MAP    = {lbl: i for i, lbl in enumerate(RATING_LABELS)}
# 5 levels: vl / l / n / h / vh  — Boehm (1981) Table 8-1
COCOMO_MULT   = {
    "RELY": [0.75, 0.88, 1.00, 1.15, 1.40],
    "DATA": [0.94, 0.94, 1.00, 1.08, 1.16],
    "CPLX": [0.70, 0.85, 1.00, 1.15, 1.30],
    "TIME": [1.00, 1.00, 1.00, 1.11, 1.30],
    "STOR": [1.00, 1.00, 1.00, 1.06, 1.21],
    "VIRT": [0.87, 0.87, 1.00, 1.15, 1.30],
    "TURN": [0.87, 0.87, 1.00, 1.07, 1.15],
    "ACAP": [1.46, 1.19, 1.00, 0.86, 0.71],
    "AEXP": [1.29, 1.13, 1.00, 0.91, 0.82],
    "PCAP": [1.42, 1.17, 1.00, 0.86, 0.70],
    "VEXP": [1.21, 1.10, 1.00, 0.90, 0.87],
    "LEXP": [1.14, 1.07, 1.00, 0.95, 0.95],
    "MODP": [1.24, 1.10, 1.00, 0.91, 0.82],
    "TOOL": [1.24, 1.10, 1.00, 0.91, 0.83],
    "SCED": [1.23, 1.08, 1.00, 1.04, 1.10],
}
# Dev mode ordinal encoding — matches train_model.py DEV_MODE_ORD
DEV_MODE_ORD  = {"Organic": 1.0, "Semi-detached": 2.0, "Embedded": 3.0}
DEV_MODE_LIST = ["Organic", "Semi-detached", "Embedded"]


# ── Pipeline ─────────────────────────────────────────────────────────────────
def inputs_to_raw(loc, ratings, dev_mode_val):
    # FEATURES[1:-1] = the 15 COCOMO cost-driver labels (RELY … SCED)
    # FEATURES[-1]   = "Dev Mode" — handled separately as ordinal
    vec = [loc] + [COCOMO_MULT[f][RATING_MAP[ratings[f]]] for f in FEATURES[1:-1]]
    vec.append(DEV_MODE_ORD[dev_mode_val])
    return np.array(vec, dtype=float).reshape(1, -1)

def preprocess(vec_raw):
    vl = np.log(np.clip(vec_raw, 1e-3, None))
    vs = scaler_X.transform(vl)
    return vs * kernel_w

def predict(vec_raw):
    vw = preprocess(vec_raw)
    ps = model.predict(vw)
    pl = scaler_y.inverse_transform(ps.reshape(-1, 1))[0, 0]
    return max(float(np.expm1(pl)), 1.0), vw

@st.cache_data(show_spinner=False)
def compute_shap(vec_raw_tuple, model_mtime):
    """
    Compute SHAP values in the ORIGINAL feature space (raw inputs → person-months).
    The full pipeline (log → MinMaxScale → data-derived weight → SVR → inverse-scale → expm1)
    is wrapped into a single function so KernelExplainer works on raw features.
    This means each SHAP value is directly interpretable as person-month contribution.

    `model_mtime` is part of the cache key only — it ensures cached SHAP values are
    invalidated when model.pkl is retrained (mirrors load_model's mtime-based busting),
    so the explanation never lags behind the prediction.
    """
    if not SHAP_AVAILABLE:
        return None, None, None
    X_raw_all = art.get("X_raw", None)
    if X_raw_all is None:
        return None, None, None
    vec_raw_arr = np.array(vec_raw_tuple).reshape(1, -1)
    np.random.seed(42)
    # Selected UI background: the 50 COCOMO-81 projects at or below 600 PM.
    # This outcome-conditioned subset makes the interactive reference explicit;
    # it is separate from the training-only backgrounds in the OOF audit.
    y_raw = art.get("y_raw_pm", None)
    if y_raw is None:
        return None, None, None
    bg = X_raw_all[y_raw <= 600.0]

    def full_pipeline(X):
        Xl = np.log(np.clip(X, 1e-3, None))
        Xs = scaler_X.transform(Xl)
        Xw = Xs * kernel_w
        ps = model.predict(Xw)
        pl = scaler_y.inverse_transform(ps.reshape(-1, 1)).ravel()
        # Match predict() exactly: local SHAP must explain the value displayed by
        # the app, including the one-person-month output floor.
        return np.maximum(np.expm1(pl), 1.0)

    exp = shap.KernelExplainer(full_pipeline, bg)
    sv = exp.shap_values(
        vec_raw_arr,
        nsamples=256,
        l1_reg=0,
        silent=True,
    )
    return sv[0], float(exp.expected_value), int(len(bg))


def local_agreement_metrics(proxy_weights, shap_values, top_k=3):
    """Descriptive agreement for one prediction; never used as research evidence."""
    return agreement_metrics(
        np.asarray(proxy_weights, dtype=float),
        np.abs(np.asarray(shap_values, dtype=float)),
        top_k=top_k,
    )


# ── Risk classification ───────────────────────────────────────────────────────
def classify_risk(effort, eaf, loc):
    """
    Risk scoring: effort size (0-2) + EAF pressure (−1 to +2) + LOC size (0-2).
    EAF < 0.90 actively reduces risk (favorable team/tools); EAF > 1.15 raises it.
    """
    score = 0
    # Effort dimension
    if effort > 300:    score += 2
    elif effort > 100:  score += 1
    # EAF dimension — favorable EAF actively lowers risk
    if eaf < 0.90:      score -= 1   # high capability/good tools: risk credit
    elif eaf > 1.15:    score += 2
    elif eaf > 1.0:     score += 1
    # LOC dimension
    if loc > 100:       score += 2
    elif loc > 30:      score += 1
    if score >= 4:      return "High",   "#C62828", "risk-high", "High"
    elif score >= 2:    return "Medium", "#BF6000", "risk-med",  "Medium"
    else:               return "Low",    "#2E7D32", "risk-low",  "Low"


# ── Executive summary ─────────────────────────────────────────────────────────
def executive_summary(effort, cal_months, team_size, eaf, loc, top_attribution, top_scenario, risk_label,
                      ai_effort=None, ai_cal=None, ai_team=None, ai_pct=None, ai_short=None):
    size_str      = ("small" if loc < 15 else "medium-sized" if loc < 100 else "large")
    effort_context = ("straightforward" if effort < 50 else "moderate" if effort < 200 else "substantial")
    lines = [
        f"This {size_str} project ({loc:.0f} KLOC) is estimated to require "
        f"<strong>{effort:.0f} person-months</strong> of effort — "
        f"a {effort_context} undertaking with <strong>{risk_label.lower()} delivery risk</strong>.",
        f"With an illustrative team of <strong>{team_size} engineers</strong>, "
        f"the projected schedule is approximately <strong>{cal_months:.0f} months</strong>. "
        f"The largest positive local SHAP attribution is <strong>{top_attribution}</strong>"
        + (f". In the one-feature sensitivity scan, the largest modeled downward difference involves <strong>{top_scenario}</strong>." if top_scenario else "."),
    ]
    if eaf < 0.85:
        lines.append("The selected inputs produce an <strong>EAF below 0.85</strong>; this is a model association, not evidence that changing one input will cause the estimate to fall.")
    elif eaf > 1.15:
        lines.append("The selected inputs produce an <strong>EAF above 1.15</strong>; this is a model association, not a causal diagnosis of delivery risk.")
    if ai_effort is not None and ai_pct and ai_pct > 1:
        lines.append(
            f"Under the <strong>{ai_short} AI multiplier scenario</strong>, the estimate is "
            f"<strong>{ai_effort:.0f} PM ({ai_pct:.0f}% below the unadjusted estimate)</strong> — "
            f"a team of {ai_team} could deliver in roughly {ai_cal:.0f} months. "
            "This post-hoc scenario is not a causal prediction from the trained model."
        )
    return " ".join(lines)


# ── Plots ────────────────────────────────────────────────────────────────────
def _style_ax(ax, title, xlabel=None):
    ax.set_title(title, fontsize=9, fontweight="normal", color="#444444", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="#999999")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#DCDCDC")
    ax.tick_params(labelsize=8, colors="#999999")
    ax.set_facecolor("white")

def plot_shap_bar(sv_pm, effort_pm):
    order  = np.argsort(np.abs(sv_pm))[::-1][:10]
    colors = ["#D97070" if sv_pm[i] > 0 else "#6B9BD2" for i in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.barh([FEATURES[i] for i in order[::-1]], [sv_pm[i] for i in order[::-1]],
            color=colors[::-1], height=0.58, edgecolor="none")
    ax.axvline(0, color="#A1A1AA", lw=0.8, ls="--", alpha=0.7)
    _style_ax(ax, f"Local SHAP Attributions  ·  {effort_pm:.0f} PM",
              "SHAP value (person-months)")
    ax.tick_params(axis="y", labelsize=8.5, colors="#27272A")
    ax.legend(handles=[
        mpatches.Patch(color="#D97070", label="Increases effort"),
        mpatches.Patch(color="#6B9BD2", label="Decreases effort"),
    ], fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    return fig

def plot_fahp_vs_shap(sv_pm):
    sn = np.abs(sv_pm); sn = sn / (sn.sum() + 1e-9)
    fn = fahp_w / fahp_w.sum()
    x, w = np.arange(len(FEATURES)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.bar(x - w/2, fn, width=w, label="Training-derived weight proxy",
           color="#A8B8D8", edgecolor="none")
    ax.bar(x + w/2, sn, width=w, label="Current prediction |SHAP|",
           color="#C8A8A8", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Normalised weight", fontsize=8, color="#71717A")
    _style_ax(ax, "Current Prediction: Weight Proxy vs Local |SHAP|")
    ax.spines["left"].set_color("#E4E4E7")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

def plot_waterfall(sv_pm, base_pm, effort_pm):
    order  = np.argsort(np.abs(sv_pm))[::-1][:8]
    labels = [FEATURES[i] for i in order]
    vals   = [sv_pm[i]    for i in order]
    running = base_pm
    starts, widths, colors = [], [], []
    for v in vals:
        starts.append(min(running, running + v))
        widths.append(abs(v))
        colors.append("#D97070" if v > 0 else "#6B9BD2")
        running += v
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.barh(labels[::-1], widths[::-1], left=starts[::-1],
            color=colors[::-1], height=0.56, edgecolor="none")
    ax.axvline(base_pm,   color="#A1A1AA", lw=1.2, ls=":",
               label=f"Baseline {base_pm:.0f} PM")
    ax.axvline(effort_pm, color="#09090B", lw=1.5,
               label=f"Prediction {effort_pm:.0f} PM")
    _style_ax(ax, "Local SHAP Attribution Waterfall", "Person-months")
    ax.tick_params(axis="y", colors="#27272A", labelsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

def plot_confidence_band(effort, q10, q90, ai_effort=None):
    """Render 80% PI band — title is shown via Streamlit, not matplotlib."""
    lo, hi = prediction_interval(effort, q10, q90, minimum=1.0)
    has_ai = ai_effort is not None and ai_effort != effort
    row_h  = 0.52
    y_base = 0.38 if has_ai else 0.0
    y_ai   = -0.65 if has_ai else None
    fig_h  = 2.8 if has_ai else 2.2
    fig, ax = plt.subplots(figsize=(7, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def _draw_band(y, center, band_lo, band_hi, color, label):
        ax.barh([y], [band_hi - band_lo], left=[band_lo], height=row_h,
                color=color, alpha=0.10, edgecolor="none")
        ax.barh([y], [band_hi - band_lo], left=[band_lo], height=row_h,
                color="none", edgecolor=color, linewidth=1.1, alpha=0.6)
        ax.plot([center], [y], marker="|", markersize=20,
                color=color, markeredgewidth=2.5, linewidth=0)
        # Endpoint labels — ABOVE the bar to avoid clashing with x-axis ticks
        ax.text(band_lo, y + row_h * 0.62, f"{band_lo:.0f}",
                ha="center", va="bottom", fontsize=7.5, color="#71717A",
                fontfamily="monospace")
        ax.text(band_hi, y + row_h * 0.62, f"{band_hi:.0f}",
                ha="center", va="bottom", fontsize=7.5, color="#71717A",
                fontfamily="monospace")
        # Row label — left of the chart (y-axis label replacement)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else band_lo * 0.55,
                y, label, ha="right", va="center",
                fontsize=8, fontweight="bold", color=color, fontfamily="sans-serif")

    # Compute xlim first so label placement is stable
    all_vals = [lo, hi, effort]
    if has_ai:
        ai_lo, ai_hi = prediction_interval(ai_effort, q10, q90, minimum=1.0)
        all_vals += [ai_lo, ai_hi, ai_effort]
    x_min = min(all_vals) * 0.60
    x_max = max(all_vals) * 1.14
    ax.set_xlim(x_min, x_max)

    _draw_band(y_base, effort, lo, hi, "#2563EB", f"Baseline  {effort:.0f} PM")
    if has_ai:
        _draw_band(y_ai, ai_effort, ai_lo, ai_hi, "#16A34A", f"With AI  {ai_effort:.0f} PM")

    ax.set_yticks([])
    ax.set_xlabel("Person-months", fontsize=8, color="#71717A", fontfamily="sans-serif")
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#E4E4E7")
    ax.tick_params(labelsize=7.5, colors="#71717A")
    ax.grid(axis="x", color="#F4F4F5", linewidth=0.7, zorder=0)
    # Add headroom above the bar so endpoint labels don't get clipped
    top_y = (y_base if not has_ai else max(y_base, y_ai or y_base)) + row_h * 1.0
    bot_y = (y_ai if has_ai else y_base) - row_h * 0.8
    ax.set_ylim(bot_y, top_y)
    plt.tight_layout(pad=1.2)
    return fig


# ── GenAI comparison chart ───────────────────────────────────────────────────
def plot_ai_comparison(effort, ai_effort, ai_lo, ai_hi, ai_label):
    modeled_difference = effort - ai_effort
    pct = modeled_difference / effort * 100
    labels = ["Baseline\n(no AI)", f"With AI\n({ai_label})"]
    values = [effort, ai_effort]
    colors = ["#3D5A80", "#4E8A6E"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(labels, values, color=colors, height=0.4,
                   edgecolor="none", alpha=0.85)
    # Uncertainty range for AI bar
    ax.barh([labels[1]], [ai_hi - ai_lo], left=[ai_lo],
            height=0.4, color="#16A34A", alpha=0.15,
            edgecolor="#4E8A6E", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(val + effort * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} PM", va="center",
                fontfamily="monospace", fontsize=9, fontweight="bold", color="#09090B")
    ax.text(ai_hi + effort * 0.02,
            bars[1].get_y() + bars[1].get_height() / 2 - 0.22,
            f"{ai_lo:.0f}–{ai_hi:.0f} range",
            va="center", fontfamily="monospace", fontsize=7.5, color="#71717A")
    ax.set_xlabel("Person-months", fontsize=8, color="#71717A", fontfamily="sans-serif")
    _style_ax(ax, f"Assumed GenAI scenario: −{modeled_difference:.0f} PM  (−{pct:.0f}%)")
    ax.set_xlim(0, effort * 1.45)
    ax.grid(axis="x", color="#F4F4F5", linewidth=0.7, zorder=0)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ── What-if ──────────────────────────────────────────────────────────────────
def whatif(base_raw, base_effort, ratings, dev_mode_val):
    """
    Evaluate pre-defined one-at-a-time rating scenarios.
    Capability/experience/practice drivers (ACAP, AEXP, PCAP, VEXP, LEXP, MODP, TOOL):
      test one rating step upward.
    Stress drivers (CPLX, RELY, TIME, STOR):
      test one rating step downward.
    Dev Mode is a project constant — not included in what-if.
    """
    scenarios = {
        "ACAP": ("Set analyst capability one level higher",     +1),
        "AEXP": ("Set application experience one level higher", +1),
        "PCAP": ("Set programmer capability one level higher",  +1),
        "VEXP": ("Set VM experience one level higher",          +1),
        "LEXP": ("Set language experience one level higher",    +1),
        "MODP": ("Set practices rating one level higher",       +1),
        "TOOL": ("Set tools rating one level higher",           +1),
        "CPLX": ("Set complexity one level lower",              -1),
        "RELY": ("Set reliability requirement one level lower", -1),
        "TIME": ("Set time constraint one level lower",         -1),
        "STOR": ("Set memory constraint one level lower",       -1),
    }
    results = []
    for feat, (label, direction) in scenarios.items():
        cur = RATING_MAP[ratings[feat]]
        nxt = cur + direction
        if 0 <= nxt < 5:
            nr = ratings.copy(); nr[feat] = RATING_LABELS[nxt]
            ne, _ = predict(inputs_to_raw(base_raw[0, 0], nr, dev_mode_val))
            difference = ne - base_effort
            if difference < -0.5:
                results.append(dict(scenario=label, feature=feat,
                    from_=RATING_LABELS[cur], to=RATING_LABELS[nxt],
                    difference=difference, pct=difference / base_effort * 100))
    return sorted(results, key=lambda r: r["difference"])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:0 0 0.8rem 0;border-bottom:1px solid #1E1E1E;margin-bottom:0.8rem;'>
      <div style='font-size:0.85rem;font-weight:600;color:#DDDDDD;letter-spacing:-0.01em;'>Software Effort Estimator</div>
      <div style='font-size:0.67rem;color:#555555;margin-top:0.1rem;font-family:monospace;letter-spacing:0.04em;'>FAHP-inspired weights &middot; SVR &middot; SHAP</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Project scale — always visible ────────────────────────────────────────
    st.markdown("<div class='sb-section'>Project Scale</div>", unsafe_allow_html=True)
    loc = st.slider("Lines of Code (KLOC)", 1, 500, 50,
                    label_visibility="collapsed")
    st.markdown(
        f"<div style='font-size:0.75rem;color:#666666;font-family:monospace;"
        f"margin-top:-0.4rem;margin-bottom:0.9rem;'>{loc} KLOC</div>",
        unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>Development Mode</div>", unsafe_allow_html=True)
    dev_mode = st.selectbox(
        "dev_mode", DEV_MODE_LIST, index=0, label_visibility="collapsed"
    )
    st.markdown(
        f"<div style='font-size:0.72rem;color:#555555;font-family:monospace;"
        f"margin-top:-0.4rem;margin-bottom:0.9rem;'>"
        f"{'Organic: a=2.4, b=1.05' if dev_mode=='Organic' else 'Semi-det: a=3.0, b=1.12' if dev_mode=='Semi-detached' else 'Embedded: a=3.6, b=1.20'}"
        f"</div>",
        unsafe_allow_html=True)

    # ── COCOMO cost drivers in collapsible groups ─────────────────────────────
    with st.expander("Product Attributes", expanded=False):
        rely = st.selectbox("RELY · Required Reliability", RATING_LABELS, index=2,
                            help="What happens if the software fails?\nVery Low = minor inconvenience · High = financial loss / SLA breach · Very High = safety-critical / regulatory")
        data = st.selectbox("DATA · Database Size",        RATING_LABELS, index=2,
                            help="Volume of data relative to codebase.\nMaps to: DB size, ETL pipeline complexity, analytics workloads, data lake scale")
        cplx = st.selectbox("CPLX · Process Complexity",   RATING_LABELS, index=2,
                            help="Algorithmic difficulty of the core logic.\nVery Low = simple CRUD · High = ML models / distributed systems · Very High = real-time AI, custom algorithms")

    with st.expander("Computer Constraints", expanded=False):
        time_ = st.selectbox("TIME · CPU Time Constraint",  RATING_LABELS, index=2,
                             help="How tight are execution-time / latency SLAs?\nNominal = standard web app · High = sub-100ms API · Very High = real-time processing, HFT")
        stor  = st.selectbox("STOR · Memory Constraint",    RATING_LABELS, index=2,
                             help="Memory-limited deployment targets.\nNominal = standard cloud VMs · High = serverless/mobile · Very High = edge devices, embedded hardware")
        virt  = st.selectbox("VIRT · VM Volatility",        RATING_LABELS, index=2,
                             help="How often does the underlying platform change?\nLow = stable cloud infra · High = frequent k8s upgrades · Very High = rapid provider / framework migrations")
        turn  = st.selectbox("TURN · Turnaround Time",      RATING_LABELS, index=2,
                             help="Dev environment feedback speed.\nMaps to: CI/CD pipeline time, PR review turnaround, build latency, local test cycle time")

    with st.expander("Team & Personnel", expanded=False):
        acap = st.selectbox("ACAP · Analyst Capability",    RATING_LABELS, index=2,
                            help="Quality of architects and senior analysts.\nMaps to: system design quality, Jira story accuracy, tech debt prevention, API contract clarity")
        aexp = st.selectbox("AEXP · App Experience",        RATING_LABELS, index=2,
                            help="Years of experience in this specific domain.\nMaps to: industry vertical experience (fintech, healthcare, logistics, e-commerce)")
        pcap = st.selectbox("PCAP · Programmer Capability", RATING_LABELS, index=2,
                            help="Developer coding quality and throughput.\nMaps to: sprint velocity, code review standards, PR merge quality, GitHub contribution metrics")
        vexp = st.selectbox("VEXP · VM Experience",         RATING_LABELS, index=2,
                            help="Platform and infrastructure familiarity.\nMaps to: AWS/GCP/Azure experience, Kubernetes, microservices, container orchestration")
        lexp = st.selectbox("LEXP · Language Experience",   RATING_LABELS, index=2,
                            help="Proficiency in the primary language and framework stack.\nMaps to: Python, TypeScript, Go, Rust, React, FastAPI, Django, Spring Boot")

    with st.expander("Project Practices", expanded=False):
        modp = st.selectbox("MODP · Modern Practices",    RATING_LABELS, index=2,
                            help="Software engineering maturity.\nMaps to: Agile/Scrum adoption, TDD coverage, code review culture, retrospective action follow-through")
        tool = st.selectbox("TOOL · Software Tools",      RATING_LABELS, index=2,
                            help="Quality and adoption of dev tooling.\nMaps to: GitHub Copilot/Cursor usage, automated testing, deployment pipelines, observability/monitoring")
        sced = st.selectbox("SCED · Schedule Constraint", RATING_LABELS, index=2,
                            help="Deadline pressure relative to nominal schedule.\nNominal = flexible · High = fixed product launch · Very High = hard regulatory / contractual deadline")

    st.markdown("<hr style='border:none;border-top:1px solid #27272A;margin:0.8rem 0;'>",
                unsafe_allow_html=True)

    # ── GenAI section ─────────────────────────────────────────────────────────
    st.markdown("<div class='sb-section'>GenAI Productivity</div>", unsafe_allow_html=True)
    ai_level = st.selectbox(
        "ai_level",
        ["None — no AI tools",
         "Some — Copilot / ChatGPT",
         "Heavy — Cursor / AI-first team"],
        index=0,
        label_visibility="collapsed",
    )

    # Study-backed multiplier ranges
    # • Peng et al. 2023 (GitHub, n=95 devs): 55% faster on isolated tasks → ~18–25% full project
    # • McKinsey 2023 (850 projects): 20–45% depending on AI adoption depth
    # • Kalliamvakou 2022 (GitHub): 26% faster for Copilot users
    AI_MULT = {
        "None — no AI tools":             1.00,
        "Some — Copilot / ChatGPT":       0.78,   # midpoint of 18–25% range
        "Heavy — Cursor / AI-first team": 0.62,   # midpoint of 30–45% range
    }
    AI_LO = {   # optimistic bound (max reduction)
        "None — no AI tools":             1.00,
        "Some — Copilot / ChatGPT":       0.75,
        "Heavy — Cursor / AI-first team": 0.55,
    }
    AI_HI = {   # conservative bound (min reduction)
        "None — no AI tools":             1.00,
        "Some — Copilot / ChatGPT":       0.82,
        "Heavy — Cursor / AI-first team": 0.70,
    }
    AI_SOURCE = {
        "None — no AI tools":             "",
        "Some — Copilot / ChatGPT":       "−18–25% · Peng et al. 2023, McKinsey 2023",
        "Heavy — Cursor / AI-first team": "−30–45% · McKinsey 2023, Kalliamvakou 2022",
    }
    ai_mult    = AI_MULT[ai_level]
    ai_mult_lo = AI_LO[ai_level]
    ai_mult_hi = AI_HI[ai_level]

    if ai_level != "None — no AI tools":
        st.markdown(
            f"<div class='sb-ai-source'>{ai_mult:.0%} estimate &nbsp;·&nbsp; "
            f"{AI_SOURCE[ai_level]}</div>",
            unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #27272A;margin:0.8rem 0;'>",
                unsafe_allow_html=True)
    if st.button("▶  Estimate Effort", width="stretch", type="primary"):
        st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    dark_mode = st.toggle("Dark mode", value=st.session_state.get("dark_mode", False),
                          key="dark_mode")

ratings = dict(
    RELY=rely, DATA=data, CPLX=cplx, TIME=time_, STOR=stor,
    VIRT=virt, TURN=turn, ACAP=acap, AEXP=aexp, PCAP=pcap,
    VEXP=vexp, LEXP=lexp, MODP=modp, TOOL=tool, SCED=sced,
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

_nested      = COCOMO_METRICS["nested_cv"]
_pred25      = _nested["pred25"]
_pred50      = _nested["pred50"]
_abl_mmre    = COCOMO_METRICS["fahp_comparison"]["unweighted_mmre"]
_fahp_imp    = COCOMO_METRICS["fahp_comparison"]["effect_pct"]
_mode_mmre   = COCOMO_METRICS["mode_metrics"]
_n93         = COCOMO_METRICS["external_nasa93"]
_n93_mmre    = _n93["mmre"]
_n93_pred25  = _n93["pred25"]
_n93_n       = _n93["n"]

st.markdown(f"""
<div class="page-header">
  <div class="page-title"> &nbsp;Software Effort Estimator</div>
  <div class="page-subtitle">Explainable software-effort research prototype &nbsp;&middot;&nbsp; FAHP-inspired weighting &nbsp;&middot;&nbsp; SVR &nbsp;&middot;&nbsp; SHAP</div>
  <div>
    <span class="stat-pill-blue">Nested-CV MMRE {_nested['mmre']:.4f}</span>
    {f'<span class="stat-pill-green">PRED(25) {_pred25*100:.0f}%</span>' if _pred25 else ''}
    <span class="stat-pill-green">n={N_81} COCOMO-81</span>
    {f'<span class="stat-pill-dark">NASA-93 MMRE {_n93_mmre:.3f}</span>' if _n93_mmre else ''}
    <span class="stat-pill-dark">Data-derived FAHP proxy</span>
  </div>
</div>
""", unsafe_allow_html=True)



# ── Research story: MMRE trajectory table ─────────────────────────────────────
_gh_nested = GITHUB_METRICS["nested_cv"]
_gh_n = GITHUB_METRICS["n"]

cocomo_row = f"<tr><td><strong>COCOMO-81</strong> (1981 benchmark)</td><td>{N_81}</td><td style='font-family:monospace;font-weight:700;color:#2E7D32;'>{_nested['mmre']:.4f}</td><td style='font-family:monospace;'>{_pred25*100:.0f}%</td><td>Nested 10×5 CV</td></tr>"
github_row = (f"<tr><td><strong>GitHub open-source</strong> (2021&#8211;present)</td>"
              f"<td>{_gh_n}</td>"
              f"<td style='font-family:monospace;font-weight:700;color:#2563EB;'>{_gh_nested['mmre']:.4f}</td>"
              f"<td style='font-family:monospace;'>{_gh_nested['pred25']*100:.0f}%</td>"
              f"<td>Nested 10×5 CV</td></tr>")
company_row = "<tr><td><strong>Your company data</strong></td><td>—</td><td style='font-family:monospace;color:#BF6000;'>not measured</td><td style='font-family:monospace;color:#BF6000;'>not measured</td><td>Requires completed-project outcomes</td></tr>"

st.markdown(f"""
<div style='background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;'>
  <div style='font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;color:#888;margin-bottom:0.6rem;'>Research Trajectory — Why Company Data Matters</div>
  <table class='data-table'>
    <thead><tr><th>Dataset</th><th>n</th><th>Nested-CV MMRE</th><th>PRED(25)</th><th>Validation</th></tr></thead>
    <tbody>{cocomo_row}{github_row}{company_row}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:0.82rem;color:#444;padding:0.55rem 0.9rem;
background:#f0f7ff;border-radius:4px;border:1px solid #c8dcf5;margin-bottom:1rem;'>
  <strong>Navigate:</strong> &nbsp; Use the <strong>sidebar on the left</strong>
  to open the <strong>GitHub Model</strong> or <strong>Your Data</strong> pages.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS  (COCOMO-81 estimator — main content)
# ══════════════════════════════════════════════════════════════════════════════
ORGANIC_MAX_KLOC = 73  # largest organic project in training set

vec_raw       = inputs_to_raw(loc, ratings, dev_mode)
effort, vec_w = predict(vec_raw)
eaf = float(np.prod([COCOMO_MULT[f][RATING_MAP[ratings[f]]] for f in COCOMO_MULT]))

if dev_mode == "Organic" and loc > ORGANIC_MAX_KLOC:
    st.warning(
        f"Largest Organic project in training data is {ORGANIC_MAX_KLOC} KLOC. "
        f"At {loc} KLOC the model is extrapolating — treat this estimate as unreliable. "
        f"Semi-detached or Embedded mode is more appropriate at this size."
    )
cal_months    = 2.5 * (effort ** 0.38)
team_size     = max(1, round(effort / cal_months))
pi_lo_abs, pi_hi_abs = prediction_interval(effort, PI_Q10, PI_Q90, minimum=1.0)
risk_label, risk_color, risk_css, risk_icon = classify_risk(effort, eaf, loc)
# GenAI-adjusted figures
ai_effort     = effort * ai_mult
ai_effort_lo  = effort * ai_mult_lo   # optimistic (max reduction)
ai_effort_hi  = effort * ai_mult_hi   # conservative (min reduction)
ai_cal        = 2.5 * (ai_effort ** 0.38)
ai_team       = max(1, round(ai_effort / ai_cal))
ai_modeled_difference = effort - ai_effort
ai_pct = ai_modeled_difference / effort * 100
ai_using      = ai_mult < 1.0
ai_short      = ai_level.split("—")[0].strip()

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cols_data = [
    (c1, "Predicted Effort",   f"{effort:.0f}",     "person-months",
     f"<span style='font-size:0.72rem;color:#57606a;'>80% range: {pi_lo_abs:.0f} – {pi_hi_abs:.0f} PM</span>"),
    (c2, "Calendar Duration",  f"{cal_months:.0f}", "months",
     f"<span style='font-size:0.72rem;color:#57606a;'>COCOMO T = 2.5 × PM⁰·³⁸</span>"),
    (c3, "Illustrative Team",  f"{team_size}",       "engineers",
     f"<span style='font-size:0.72rem;color:#57606a;'>{effort:.0f} PM ÷ {cal_months:.0f} mo</span>"),
    (c4, "Effort Adj. Factor", f"{eaf:.3f}",         "combined EAF",
     f"<span style='font-size:0.72rem;color:#57606a;'>{'< 1 = favorable' if eaf < 1 else ('= 1 = average' if abs(eaf-1)<0.01 else '> 1 = challenging')}</span>"),
    (c5, "Delivery Risk",      risk_label,           "composite score",
     f"<span style='font-size:0.71rem;color:#AAAAAA;'>size &middot; EAF &middot; complexity</span>"),
]
for col, label, value, unit, sub in cols_data:
    if label == "Delivery Risk":
        risk_style = f"color:{risk_color};font-size:1.65rem;"
    elif label == "Effort Adj. Factor":
        risk_style = "font-size:1.9rem;"
    else:
        risk_style = ""
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='{risk_style}'>{value}</div>
        <div class='metric-unit'>{unit}</div>
        <div class='metric-range'>{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Confidence band ────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>80% Prediction Interval &nbsp;·&nbsp; empirical, nested outer folds</div>", unsafe_allow_html=True)
_fig_cb = plot_confidence_band(effort, PI_Q10, PI_Q90,
          ai_effort=ai_effort if ai_using else None)
_ = st.pyplot(_fig_cb, width="stretch")
plt.close(_fig_cb)

# ── Executive Summary ─────────────────────────────────────────────────────────
if SHAP_AVAILABLE:
    with st.spinner("Computing SHAP values…"):
        sv_pm, base_pm, local_bg_n = compute_shap(tuple(vec_raw.ravel().tolist()), _mtime)
    if sv_pm is None:
        st.error(
            "**Interactive SHAP inputs are missing from model.pkl.** "
            "Re-run `python train_model.py` before using this page.", icon="🚨"
        )
        st.stop()
    top_pos_feats = [FEATURES[i] for i in np.argsort(sv_pm)[::-1] if sv_pm[i] > 0]
else:
    sv_pm, base_pm, local_bg_n = None, None, None
    top_pos_feats = []

wi_results    = whatif(vec_raw, effort, ratings, dev_mode)
top_attribution = top_pos_feats[0] if top_pos_feats else "project size"
top_scenario    = wi_results[0]["feature"] if wi_results else None

summary_html = executive_summary(
    effort, cal_months, team_size, eaf, loc, top_attribution, top_scenario, risk_label,
    ai_effort=ai_effort if ai_using else None,
    ai_cal=ai_cal, ai_team=ai_team, ai_pct=ai_pct, ai_short=ai_short,
)
st.markdown(f"""
<div class="exec-summary">
  <div class="exec-summary-label">Executive Summary</div>
  <div class="exec-summary-text">{summary_html}</div>
</div>""", unsafe_allow_html=True)

# ── SHAP tabs ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Cross-Fitted Explainability Evidence</div>", unsafe_allow_html=True)

_primary = EXPLAINABILITY["primary_alignment"]
_primary_repeats = EXPLAINABILITY["primary_alignment_across_repeats"]
_primary_bootstrap = EXPLAINABILITY["primary_alignment_project_bootstrap"]
_rank_stability = EXPLAINABILITY["unweighted_global_ranking_stability"]
_local_stability = EXPLAINABILITY["unweighted_local_stability"]
_weighted_desc = EXPLAINABILITY["weighted_alignment_descriptive"]
_weighted_repeats = EXPLAINABILITY["weighted_alignment_across_repeats_descriptive"]
_protocol = EXPLAINABILITY["protocol"]
_permutation = EXPLAINABILITY["feature_label_permutation_null"]
_numerical = EXPLAINABILITY["numerical_sensitivity"]
_proxy_diagnostics = EXPLAINABILITY["weighting_proxy"]["full_data_diagnostics"]
_numerical_main = next(
    row for row in _numerical["sample_size_results"]
    if row["nsamples"] == _protocol["nsamples"]
)

ea1, ea2, ea3, ea4 = st.columns(4)
_evidence_cards = [
    (ea1, "Primary agreement ρ", _primary["spearman_rho"],
     "proxy vs unweighted OOF |SHAP|",
     f"conditional project interval {_primary_bootstrap['metrics']['spearman_rho']['lower']:.3f}–{_primary_bootstrap['metrics']['spearman_rho']['upper']:.3f}"),
    (ea2, "Joint rank stability ρ", _rank_stability["spearman_rho"]["median"],
     "splits, models, backgrounds, seeds vary",
     f"IQR {_rank_stability['spearman_rho']['q25']:.3f}–{_rank_stability['spearman_rho']['q75']:.3f}"),
    (ea3, "Joint local stability", _local_stability["cosine_similarity"]["median"],
     "same project across repeated pipeline",
     f"IQR {_local_stability['cosine_similarity']['q25']:.3f}–{_local_stability['cosine_similarity']['q75']:.3f}"),
    (ea4, "Weighted agreement ρ*", _weighted_desc["spearman_rho"],
     "descriptive weighted-model result",
     f"repeat median {_weighted_repeats['spearman_rho']['median']:.3f} · IQR {_weighted_repeats['spearman_rho']['q25']:.3f}–{_weighted_repeats['spearman_rho']['q75']:.3f}"),
]
for _col, _label, _value, _unit, _range in _evidence_cards:
    _col.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{_label}</div>
        <div class='metric-value' style='font-size:1.9rem;'>{_value:.3f}</div>
        <div class='metric-unit'>{_unit}</div>
        <div class='metric-range'>{_range}</div>
    </div>""", unsafe_allow_html=True)

_feature_rows = sorted(
    EXPLAINABILITY["feature_table"], key=lambda row: row["unweighted_shap_rank"]
)[:5]
_feature_rows_html = "".join(
    f"<tr><td><strong>{row['feature']}</strong></td>"
    f"<td style='font-family:monospace;'>#{row['proxy_rank']} &nbsp;({row['mean_weighting_proxy']*100:.1f}%)</td>"
    f"<td style='font-family:monospace;'>#{row['unweighted_shap_rank']} &nbsp;({row['unweighted_oof_shap_reliance']*100:.1f}%)</td>"
    f"<td style='font-family:monospace;'>{row['unweighted_shap_rank_iqr'][0]:.1f}–{row['unweighted_shap_rank_iqr'][1]:.1f}</td>"
    f"<td style='font-family:monospace;'>{row['unweighted_top_k_frequency']*100:.0f}%</td>"
    f"<td style='font-family:monospace;'>#{row['weighted_shap_rank']} &nbsp;({row['weighted_oof_shap_reliance']*100:.1f}%)</td></tr>"
    for row in _feature_rows
)
st.markdown(f"""
<table class='data-table' style='margin-top:0.8rem;'>
  <thead><tr><th>Top feature</th><th>Weight proxy</th><th>Unweighted OOF |SHAP|</th>
    <th>Repeat rank IQR</th><th>Top-{_protocol['top_k']} frequency</th><th>Weighted |SHAP|*</th></tr></thead>
  <tbody>{_feature_rows_html}</tbody>
</table>
<div class='explain-note' style='margin-top:0.65rem;'>
  <div class='explain-label'>How to read this evidence</div>
  <div class='explain-text'>
    The primary comparison is the <strong>training-derived weighting proxy versus out-of-fold
    SHAP reliance from an unweighted SVR</strong>. It avoids measuring a model against weights
    already injected into that model. The weighted-model agreement marked * is reported separately
    and is descriptive. The primary repeat-level ρ median is
    {_primary_repeats['spearman_rho']['median']:.3f} (IQR
    {_primary_repeats['spearman_rho']['q25']:.3f}–{_primary_repeats['spearman_rho']['q75']:.3f});
    its {_primary_bootstrap['coverage']:.0%} conditional project-resampling percentile interval uses
    {_primary_bootstrap['resamples']:,} resamples without refitting the pipeline. Results use
    {_protocol['repeats']} repeated {_protocol['outer_splits']}-fold
    outer evaluations with {_protocol['inner_splits']}-fold inner selection; only outer-test projects
    are explained, and every SHAP background contains outer-training projects only. Agreement and
    stability jointly reflect split, model, background, and explainer-seed changes. They describe model behaviour—they do not establish causal feature effects or validate the
    data-derived proxy as expert judgment. Secondary pooled agreement: Kendall τ<sub>b</sub>
    {_primary['kendall_tau_b']:.3f}, Jensen–Shannon similarity
    {_primary['jensen_shannon_similarity']:.3f}, and top-{_primary['top_k']} Jaccard
    {_primary['top_k_jaccard']:.3f}. In the narrow {_permutation['permutations']:,}-shuffle
    feature-label benchmark, the empirical two-sided p-value for ρ is
    {_permutation['empirical_p_values']['spearman_rho']['empirical_p']:.4f}, while top-{_primary['top_k']}
    overlap is not unusual (upper-tail p={_permutation['empirical_p_values']['top_k_jaccard']['empirical_p']:.4f}).
    KernelSHAP uses <code>l1_reg=0</code>, so all 17 features enter the solution. With the model, rows, and
    background fixed, {_protocol['nsamples']} samples reproduce the 1,024-sample reference ranking at
    ρ={_numerical_main['global_agreement_to_1024_reference']['spearman_rho']:.3f}; median local cosine is
    {_numerical_main['local_cosine_to_1024_reference']['median']:.4f}. This numerical audit is not predictive evidence.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-header'>Current Prediction · Local Explanation</div>", unsafe_allow_html=True)

if not SHAP_AVAILABLE:
    st.info(
        "Interactive SHAP for the current inputs is unavailable in this deployment. "
        "The stored cross-fitted research evidence above remains available.", icon="ℹ️"
    )
else:
    tab1, tab2, tab3 = st.tabs(["Local Contributions", "Local Weight Comparison", "Waterfall"])
    with tab1:
        _fig_bar = plot_shap_bar(sv_pm, effort)
        _ = st.pyplot(_fig_bar, width="stretch"); plt.close(_fig_bar)
        st.markdown(f"""<div class='insight-note' style='margin-top:0.4rem;'>
            <strong>Local, descriptive explanation:</strong> Each value is relative to the model's
            expected prediction over the selected <strong>{local_bg_n}-project background</strong>:
            the COCOMO-81 projects with recorded effort ≤ 600 person-months. This outcome-conditioned
            background is used only for the interactive display; it is not the training-only background
            used for the cross-fitted evidence above. SHAP attributes this prediction—it does not show
            what would happen if a feature were changed.
        </div>""", unsafe_allow_html=True)
    with tab2:
        _fig_fvs = plot_fahp_vs_shap(sv_pm)
        _ = st.pyplot(_fig_fvs, width="stretch"); plt.close(_fig_fvs)
        _local_agreement = local_agreement_metrics(fahp_w, sv_pm, top_k=_protocol["top_k"])
        st.markdown(f"""<div class='insight-note' style='margin-top:0.4rem;'>
            <strong>Descriptive for these inputs only:</strong>
            Spearman ρ = {_local_agreement['spearman_rho']:.3f} &nbsp;·&nbsp;
            Kendall τ<sub>b</sub> = {_local_agreement['kendall_tau_b']:.3f} &nbsp;·&nbsp;
            Jensen–Shannon similarity = {_local_agreement['jensen_shannon_similarity']:.3f} &nbsp;·&nbsp;
            top-{_local_agreement['top_k']} Jaccard = {_local_agreement['top_k_jaccard']:.3f}.<br>
            This single-project comparison is not part of the out-of-fold research result and should
            not be interpreted as a validation score.
        </div>""", unsafe_allow_html=True)
    with tab3:
        _fig_wf = plot_waterfall(sv_pm, base_pm, effort)
        _ = st.pyplot(_fig_wf, width="stretch"); plt.close(_fig_wf)

# ── Insights ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Current-Prediction Attributions</div>", unsafe_allow_html=True)
if sv_pm is not None:
    top_pos = [(FEATURES[i], sv_pm[i]) for i in np.argsort(sv_pm)[::-1] if sv_pm[i] > 0][:3]
    top_neg = [(FEATURES[i], sv_pm[i]) for i in np.argsort(sv_pm)       if sv_pm[i] < 0][:2]
else:
    top_pos, top_neg = [], []

def _rating_tag(feat):
    """Return a short human-readable tag showing the current setting for a feature."""
    if feat == "LOC (KLOC)":
        return f"{loc:.0f} KLOC"
    if feat == "Dev Mode":
        return dev_mode
    return ratings.get(feat, "")

ic1, ic2 = st.columns(2)
with ic1:
    st.markdown("**Positive local attributions**")
    for feat, val in top_pos:
        tag = _rating_tag(feat)
        tag_html = f" <span style='font-size:0.75rem;color:#999;font-weight:400;'>({tag})</span>" if tag else ""
        st.markdown(f"""<div class='insight-box insight-up'>
            <strong>{feat}</strong>{tag_html}<br>
            <span style='font-family:monospace;'>+{val:.1f} PM</span> relative to the selected background
        </div>""", unsafe_allow_html=True)
with ic2:
    st.markdown("**Negative local attributions**")
    for feat, val in top_neg:
        tag = _rating_tag(feat)
        tag_html = f" <span style='font-size:0.75rem;color:#999;font-weight:400;'>({tag})</span>" if tag else ""
        st.markdown(f"""<div class='insight-box insight-down'>
            <strong>{feat}</strong>{tag_html}<br>
            <span style='font-family:monospace;'>{val:.1f} PM</span> relative to the selected background
        </div>""", unsafe_allow_html=True)

# ── FAHP vs SHAP alignment ─────────────────────────────────────────────────────

# ── What-if ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>One-at-a-Time Sensitivity Scenarios</div>", unsafe_allow_html=True)
if wi_results:
    wi_rows = "".join(
        f"<tr><td>{r['scenario']}</td>"
        f"<td style='font-family:monospace;font-size:0.77rem;color:#555;'>{r['feature']}</td>"
        f"<td>{r['from_']} &rarr; {r['to']}</td>"
        f"<td style='font-family:monospace;font-weight:600;color:#2E7D32;'>{r['difference']:+.1f}</td>"
        f"<td style='font-family:monospace;color:#2E7D32;'>{r['pct']:+.1f}%</td></tr>"
        for r in wi_results
    )
    st.markdown(f"""
    <table class='data-table'>
      <thead><tr>
        <th>Scenario</th><th>Feature</th><th>Input change</th><th>Model Δ PM</th><th>Model Δ %</th>
      </tr></thead>
      <tbody>{wi_rows}</tbody>
    </table>""", unsafe_allow_html=True)
    b = wi_results[0]
    st.markdown(f"""<div class='wi-callout'>
        <strong>Largest modeled one-feature difference:</strong> {b['scenario']}
        &nbsp;&middot;&nbsp; {b['feature']}: {b['from_']} &rarr; {b['to']}
        &nbsp;&middot;&nbsp; model Δ <strong>{b['difference']:+.0f} PM ({b['pct']:+.1f}%)</strong>
        <br><span style='font-size:0.78rem;opacity:0.8;'>
        Each row changes one encoded input while holding all others fixed. It is a sensitivity
        calculation, not evidence that making the change would cause this outcome or a recommendation
        that the scenario is feasible, safe, or cost-effective.
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.info("None of the pre-defined one-step sensitivity scenarios lowered the model estimate by more than 0.5 PM.")

# ── GenAI sensitivity scenario ────────────────────────────────────────────────
st.markdown("<div class='section-header'>Post-hoc GenAI Sensitivity Scenario</div>", unsafe_allow_html=True)
if ai_using:
    ga1, ga2 = st.columns([1.5, 1])
    with ga1:
        _fig_ai = plot_ai_comparison(effort, ai_effort, ai_effort_lo, ai_effort_hi, ai_short)
        _ = st.pyplot(_fig_ai, width="stretch"); plt.close(_fig_ai)
    with ga2:
        ai_pct_lo = (1 - ai_mult_hi) * 100
        ai_pct_hi = (1 - ai_mult_lo) * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Assumed Adjustment</div>
            <div class='metric-value' style='color:#16A34A;font-size:1.85rem;'>{ai_pct:.0f}%</div>
            <div class='metric-unit'>below unadjusted model estimate</div>
            <div class='metric-range'>{ai_pct_lo:.0f}–{ai_pct_hi:.0f}% illustrative range</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>AI-Adjusted Estimate</div>
            <div class='metric-value' style='font-size:1.85rem;'>{ai_effort:.0f}</div>
            <div class='metric-unit'>person-months</div>
            <div class='metric-range'>{ai_effort_lo:.0f}–{ai_effort_hi:.0f} PM range</div>
        </div>""", unsafe_allow_html=True)
    source_text = AI_SOURCE[ai_level]
    st.markdown(f"""<div class='explain-note' style='margin-top:0.8rem;'>
        <div class='explain-label'>Source</div>
        <div class='explain-text'>
            Illustrative multipliers derived from published task studies and industry surveys:
            <strong>{source_text}</strong>. They are applied after prediction and were not learned or
            validated on this dataset. Extending task-level findings to total project effort is an
            exploratory sensitivity calculation, not a causal estimate or a novelty claim.
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class='explain-note'>
        <div class='explain-label'>GenAI adjustment disabled</div>
        <div class='explain-text'>
            Select a GenAI adoption level in the sidebar to apply an illustrative post-hoc
            multiplier sourced from published task studies and industry reports. The resulting
            scenario is not learned by this model and should not be treated as a causal forecast.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Validation Evidence ───────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Validation Evidence</div>", unsafe_allow_html=True)

ve1, ve2 = st.columns(2)

with ve1:
    st.markdown("**Ablation: Does the data-derived weighting proxy change error?**")
    if _abl_mmre and _fahp_imp:
        st.markdown(f"""<table class='data-table'>
          <thead><tr><th>Model</th><th>Nested-CV MMRE</th><th>PRED(25)</th></tr></thead>
          <tbody>
            <tr><td>Unweighted SVR (baseline)</td>
                <td style='font-family:monospace;color:#C62828;font-weight:600;'>{_abl_mmre:.4f}</td>
                <td style='color:#888;'>—</td></tr>
            <tr><td><strong>SVR with data-derived weights</strong></td>
                <td style='font-family:monospace;color:#2E7D32;font-weight:600;'>{_nested['mmre']:.4f}</td>
                <td style='font-family:monospace;color:#2E7D32;font-weight:600;'>{f"{_pred25*100:.0f}%" if _pred25 else "—"}</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='wi-callout' style='margin-top:0.5rem;'>
            In matched nested validation, the weighting proxy changes MMRE by <strong>{_fahp_imp:.1f}%</strong>
            — from {_abl_mmre:.4f} to {_nested['mmre']:.4f}. This is a comparison,
            not proof that the weighting method will improve every future dataset.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Ablation data not available — re-run train_model.py.")

with ve2:
    st.markdown("**Cross-Dataset Generalization (NASA-93)**")
    if _n93_mmre is not None:
        st.markdown(f"""<table class='data-table'>
          <thead><tr><th>Evaluation</th><th>Dataset</th><th>n</th><th>MMRE</th><th>PRED(25)</th></tr></thead>
          <tbody>
            <tr><td>Nested 10×5 CV</td><td>COCOMO-81</td><td>{N_81}</td>
                <td style='font-family:monospace;font-weight:600;color:#2E7D32;'>{_nested['mmre']:.4f}</td>
                <td style='font-family:monospace;'>{f"{_pred25*100:.0f}%" if _pred25 else "—"}</td></tr>
            <tr><td>Out-of-sample test</td><td>NASA-93</td><td>{_n93_n}</td>
                <td style='font-family:monospace;font-weight:600;color:#BF6000;'>{_n93_mmre:.4f}</td>
                <td style='font-family:monospace;'>{f"{_n93_pred25*100:.0f}%" if _n93_pred25 else "—"}</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='insight-note' style='margin-top:0.5rem;'>
            Model trained on COCOMO-81 (63 projects) and tested on
            NASA-93 (93 entirely different NASA projects) — no data leakage.
            The external MMRE of {_n93_mmre:.3f} measures transfer to a separate dataset;
            it does not guarantee accuracy for modern commercial projects.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("NASA-93 validation not available — re-run train_model.py.")

# Per-mode breakdown
if _mode_mmre:
    st.markdown("**Per Development Mode — MMRE Breakdown**")
    mode_rows = ""
    mode_labels = {"organic": "Organic", "semidetached": "Semi-detached", "embedded": "Embedded"}
    mode_desc   = {
        "organic":      "Small team, familiar domain, flexible requirements",
        "semidetached": "Mixed experience, moderate constraints",
        "embedded":     "Hard real-time, safety-critical, tight hardware",
    }
    for k, label in mode_labels.items():
        if k in _mode_mmre:
            m = _mode_mmre[k]
            colour = "#2E7D32" if m["mmre"] < 0.30 else ("#BF6000" if m["mmre"] < 0.40 else "#C62828")
            mode_rows += (
                f"<tr><td><strong>{label}</strong></td>"
                f"<td style='color:#666;font-size:0.79rem;'>{mode_desc[k]}</td>"
                f"<td style='text-align:center;'>{m['n']}</td>"
                f"<td style='font-family:monospace;font-weight:600;color:{colour};'>{m['mmre']:.4f}</td>"
                f"<td style='font-family:monospace;'>{m['pred25']*100:.0f}%</td></tr>"
            )
    st.markdown(f"""<table class='data-table'>
      <thead><tr><th>Mode</th><th>Project Type</th><th>n</th><th>Nested-fold MMRE</th><th>PRED(25)</th></tr></thead>
      <tbody>{mode_rows}</tbody>
    </table>""", unsafe_allow_html=True)

# ── Model Limitations ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Validity &amp; Limitations</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='explain-note' style='border-left-color:#BF6000;'>
  <div class='explain-label' style='color:#BF6000;'>Scope of validity — read before citing results</div>
  <div class='explain-text'>
    <strong>Training data:</strong> {N_81} COCOMO-81 projects (1980s defense/aerospace software).
    The model has no exposure to React, microservices, CI/CD,
    or cloud-native development. Results should be treated as directional estimates, not budget commitments.<br><br>
    <strong>Method:</strong> Feature weights are derived from project-level FAHP quantities and Spearman
    association, so they are a data-derived proxy rather than independent expert judgments. The solver is
    sklearn SVR, not the weighted-kernel LSSVM studied by Sehra et al. (2019). This implementation is an
    adaptation inspired by that work, not a faithful reproduction of its complete method.<br><br>
    <strong>Proxy diagnostic:</strong> On the full dataset, Chang extent analysis assigns non-zero
    project weights to {_proxy_diagnostics['effort_project_weights_nonzero']}/{_proxy_diagnostics['project_count']}
    effort entries and {_proxy_diagnostics['loc_project_weights_nonzero']}/{_proxy_diagnostics['project_count']}
    LOC entries; the effort pairwise consistency ratio is
    {_proxy_diagnostics['effort_pairwise_consistency_ratio']:.3f}. This sparsity requires sensitivity
    analysis, and a confirmatory extension still needs frozen expert judgments or a documented new panel.<br><br>
    <strong>GenAI adjustment:</strong> Post-hoc multipliers from external studies, not trained into the model.
    Applied to total effort as an illustrative sensitivity scenario; the actual effect may differ
    and must be measured independently on completed projects.
    The <strong>80% interval</strong> converts the 10th and 90th percentiles of pure nested
    outer-fold signed relative errors using actual = prediction / (1 + error). Roughly 1-in-5
    comparable projects may fall outside this empirical range.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Methodology expander ───────────────────────────────────────────────────────
with st.expander("Methodology — how this works", expanded=False):
    st.markdown("""
**Pipeline overview**

1. **Training-derived weighting proxy** — Chang (1996) extent analysis builds project-level pairwise matrices from the effort/LOC distributions of all 63 training projects. Feature weights are then derived by Spearman association with those project weights. This is not an independent expert panel.

2. **Weighted-feature SVR adaptation** — after log transformation and scaling, inputs are mapped by `S = diag(√θ)` before sklearn's RBF SVR. Equivalently, `K_θ(xₖ,xₗ) = exp(−γ Σⱼ θⱼ(xₖⱼ−xₗⱼ)²)`. This is an SVR adaptation of Sehra et al. (2019), not their LSSVM solver.

3. **Nested validation** — Ten outer folds are used only for evaluation. Within each outer-training partition, five-fold inner CV selects hyperparameters and all scaling/weight-proxy fitting is repeated without seeing the outer-test rows. The deployment model is selected separately after evaluation.

4. **Empirical 80% PI** — 10th/90th percentile of pure outer-fold signed relative errors. No parametric assumption.

5. **Cross-fitted SHAP evidence** — ten repeated 5-fold outer evaluations with 3-fold inner selection explain only outer-test projects. Each model-agnostic KernelSHAP background contains all corresponding outer-training projects. The primary agreement analysis compares the training-derived proxy with unweighted-model out-of-fold |SHAP|; weighted-model agreement is descriptive.

6. **Interactive local SHAP** — KernelExplainer uses the selected 50 COCOMO-81 projects with recorded effort ≤ 600 PM as its outcome-conditioned display background (`nsamples=256`, `l1_reg=0`, seed=42). Disabling L1 feature selection keeps all 17 features in the weighted least-squares solution. It explains the displayed prediction in person-months and is separate from the cross-fitted evidence.

7. **GenAI Adjustment** — Post-hoc multiplier from Peng et al. 2023, McKinsey 2023, Kalliamvakou 2022. Not trained into the model.
""")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
n_81_val = art.get("n_cocomo81", 63)
n_93_val = art.get("n_nasa93", 0)
mmre_val = _nested["mmre"]
st.markdown(f"""
<div class='app-footer'>
  <div class='footer-left'>
    <strong style='color:#333;'>Software Effort Estimator</strong> &nbsp;&middot;&nbsp;
    Research demo &nbsp;&middot;&nbsp; Mitacs Internship Proposal<br>
    Training data: COCOMO-81 (n={n_81_val}) &nbsp;&middot;&nbsp;
    FAHP-inspired SVR adaptation (Sehra et al. 2019) &nbsp;&middot;&nbsp;
    Nested-CV MMRE&nbsp;{mmre_val:.4f} &nbsp;&middot;&nbsp; residual q10/q90&nbsp;[{PI_Q10*100:+.0f}%,&nbsp;{PI_Q90*100:+.0f}%]
  </div>
  <div class='footer-right'>v0.2-demo</div>
</div>
""", unsafe_allow_html=True)
