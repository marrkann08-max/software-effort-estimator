import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os

from app_metrics import model_metrics
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
    font-family: "JetBrains Mono", monospace; font-size: 2.6rem; font-weight: 600;
    color: #111111; line-height: 1; letter-spacing: -0.03em;
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
    The full pipeline (log → MinMaxScale → FAHP-weight → SVR → inverse-scale → expm1)
    is wrapped into a single function so KernelExplainer works on raw features.
    This means each SHAP value is directly interpretable as person-month contribution.

    `model_mtime` is part of the cache key only — it ensures cached SHAP values are
    invalidated when model.pkl is retrained (mirrors load_model's mtime-based busting),
    so the explanation never lags behind the prediction.
    """
    if not SHAP_AVAILABLE:
        return None, None
    X_raw_all = art.get("X_raw", None)
    if X_raw_all is None:
        return None, None
    vec_raw_arr = np.array(vec_raw_tuple).reshape(1, -1)
    np.random.seed(42)
    # Representative background: projects <= 600 PM (80th percentile).
    # Excludes 13 defence mega-programs (up to 11,400 PM) that skew the arithmetic
    # mean to 620 PM — not representative of modern commercial software.
    # Gives base value ~116 PM (near the geometric mean of the distribution).
    y_raw = art.get("y_raw_pm", None)
    if y_raw is not None:
        bg = X_raw_all[y_raw <= 600.0]
    else:
        bg = X_raw_all

    def full_pipeline(X):
        Xl = np.log(np.clip(X, 1e-3, None))
        Xs = scaler_X.transform(Xl)
        Xw = Xs * kernel_w
        ps = model.predict(Xw)
        pl = scaler_y.inverse_transform(ps.reshape(-1, 1)).ravel()
        return np.expm1(pl)

    exp = shap.KernelExplainer(full_pipeline, bg)
    sv  = exp.shap_values(vec_raw_arr, nsamples=150, silent=True)
    return sv[0], float(exp.expected_value)


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
def executive_summary(effort, cal_months, team_size, eaf, loc, top_driver, top_saver, risk_label,
                      ai_effort=None, ai_cal=None, ai_team=None, ai_pct=None, ai_short=None):
    size_str      = ("small" if loc < 15 else "medium-sized" if loc < 100 else "large")
    effort_context = ("straightforward" if effort < 50 else "moderate" if effort < 200 else "substantial")
    lines = [
        f"This {size_str} project ({loc:.0f} KLOC) is estimated to require "
        f"<strong>{effort:.0f} person-months</strong> of effort — "
        f"a {effort_context} undertaking with <strong>{risk_label.lower()} delivery risk</strong>.",
        f"With a recommended team of <strong>{team_size} engineers</strong>, "
        f"the projected schedule is approximately <strong>{cal_months:.0f} months</strong>. "
        f"The largest effort driver is <strong>{top_driver}</strong>"
        + (f", while the best way to reduce cost is to improve <strong>{top_saver}</strong>." if top_saver else "."),
    ]
    if eaf < 0.85:
        lines.append("Project conditions are <strong>favorable</strong>: high team capability and good tooling are already significantly reducing the baseline estimate.")
    elif eaf > 1.15:
        lines.append("Project conditions are <strong>challenging</strong>: several cost drivers are pushing the estimate above the baseline — address these early to avoid overruns.")
    if ai_effort is not None and ai_pct and ai_pct > 1:
        lines.append(
            f"With <strong>{ai_short} AI tool usage</strong>, the estimate drops to "
            f"<strong>{ai_effort:.0f} PM ({ai_pct:.0f}% reduction)</strong> — "
            f"a team of {ai_team} could deliver in roughly {ai_cal:.0f} months. "
            f"Quantifying this gap (COCOMO TOOL tops out at 17%; modern AI tools exceed that) "
            f"is the novel contribution of this research."
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
    _style_ax(ax, f"Feature Contributions  ·  {effort_pm:.0f} PM",
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
    ax.bar(x - w/2, fn, width=w, label="FAHP prior weight",
           color="#A8B8D8", edgecolor="none")
    ax.bar(x + w/2, sn, width=w, label="SHAP posterior importance",
           color="#C8A8A8", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Normalised weight", fontsize=8, color="#71717A")
    _style_ax(ax, "FAHP Prior vs SHAP Posterior")
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
    _style_ax(ax, "Cumulative Contribution Waterfall", "Person-months")
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
    saving = effort - ai_effort
    pct    = saving / effort * 100
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
    _style_ax(ax, f"GenAI saves {saving:.0f} PM  ({pct:.0f}% reduction)")
    ax.set_xlim(0, effort * 1.45)
    ax.grid(axis="x", color="#F4F4F5", linewidth=0.7, zorder=0)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ── What-if ──────────────────────────────────────────────────────────────────
def whatif(base_raw, base_effort, ratings, dev_mode_val):
    """
    For each action, compute effort if we move one rating step in the beneficial direction.
    Capability/experience/practice drivers (ACAP, AEXP, PCAP, VEXP, LEXP, MODP, TOOL):
      higher rating → lower multiplier → less effort, so direction = +1 (Nominal→High).
    Stress drivers (CPLX, RELY, TIME, STOR):
      lower rating → lower multiplier → less effort, so direction = -1 (Nominal→Low).
    Dev Mode is a project constant — not included in what-if.
    """
    improvable = {
        # Capability & practice drivers: improve = move UP the scale (+1)
        "ACAP": ("Increase analyst capability",      +1),
        "AEXP": ("Increase application experience",  +1),
        "PCAP": ("Increase programmer capability",   +1),
        "VEXP": ("Increase VM experience",           +1),
        "LEXP": ("Increase language experience",     +1),
        "MODP": ("Improve modern practices",         +1),
        "TOOL": ("Adopt better tools",               +1),
        # Stress drivers: reduce = move DOWN the scale (-1)
        "CPLX": ("Reduce complexity",                -1),
        "RELY": ("Lower reliability requirement",    -1),
        "TIME": ("Relax time constraint",            -1),
        "STOR": ("Relax memory constraint",          -1),
    }
    results = []
    for feat, (label, direction) in improvable.items():
        cur = RATING_MAP[ratings[feat]]
        nxt = cur + direction
        if 0 <= nxt < 5:
            nr = ratings.copy(); nr[feat] = RATING_LABELS[nxt]
            ne, _ = predict(inputs_to_raw(base_raw[0, 0], nr, dev_mode_val))
            saving = base_effort - ne
            if saving > 0.5:
                results.append(dict(action=label, feature=feat,
                    from_=RATING_LABELS[cur], to=RATING_LABELS[nxt],
                    saving=saving, pct=saving / base_effort * 100))
    return sorted(results, key=lambda r: r["saving"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:0 0 0.8rem 0;border-bottom:1px solid #1E1E1E;margin-bottom:0.8rem;'>
      <div style='font-size:0.85rem;font-weight:600;color:#DDDDDD;letter-spacing:-0.01em;'>Software Effort Estimator</div>
      <div style='font-size:0.67rem;color:#555555;margin-top:0.1rem;font-family:monospace;letter-spacing:0.04em;'>FAHP &middot; SVR &middot; SHAP</div>
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
  <div class="page-subtitle">Explainable AI for software project planning &nbsp;&middot;&nbsp; FAHP &nbsp;&middot;&nbsp; Weighted-Kernel SVR &nbsp;&middot;&nbsp; SHAP</div>
  <div>
    <span class="stat-pill-blue">Nested-CV MMRE {_nested['mmre']:.4f}</span>
    {f'<span class="stat-pill-green">PRED(25) {_pred25*100:.0f}%</span>' if _pred25 else ''}
    <span class="stat-pill-green">n={N_81} COCOMO-81</span>
    {f'<span class="stat-pill-dark">NASA-93 MMRE {_n93_mmre:.3f}</span>' if _n93_mmre else ''}
    <span class="stat-pill-dark">Chang 1996 FAHP</span>
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
ai_saving     = effort - ai_effort
ai_pct        = ai_saving / effort * 100
ai_using      = ai_mult < 1.0
ai_short      = ai_level.split("—")[0].strip()

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cols_data = [
    (c1, "Predicted Effort",   f"{effort:.0f}",     "person-months",
     f"<span style='font-size:0.72rem;color:#57606a;'>80% range: {pi_lo_abs:.0f} – {pi_hi_abs:.0f} PM</span>"),
    (c2, "Calendar Duration",  f"{cal_months:.0f}", "months",
     f"<span style='font-size:0.72rem;color:#57606a;'>COCOMO T = 2.5 × PM⁰·³⁸</span>"),
    (c3, "Recommended Team",   f"{team_size}",       "engineers",
     f"<span style='font-size:0.72rem;color:#57606a;'>{effort:.0f} PM ÷ {cal_months:.0f} mo</span>"),
    (c4, "Effort Adj. Factor", f"{eaf:.3f}",         "combined EAF",
     f"<span style='font-size:0.72rem;color:#57606a;'>{'< 1 = favorable' if eaf < 1 else ('= 1 = average' if abs(eaf-1)<0.01 else '> 1 = challenging')}</span>"),
    (c5, "Delivery Risk",      risk_label,           "composite score",
     f"<span style='font-size:0.71rem;color:#AAAAAA;'>size &middot; EAF &middot; complexity</span>"),
]
for col, label, value, unit, sub in cols_data:
    risk_style = f"color:{risk_color};" if label == "Delivery Risk" else ""
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
        sv_pm, base_pm = compute_shap(tuple(vec_raw.ravel().tolist()), _mtime)
    top_pos_feats = [FEATURES[i] for i in np.argsort(sv_pm)[::-1] if sv_pm[i] > 0]
else:
    sv_pm, base_pm = None, None
    top_pos_feats = []

wi_results    = whatif(vec_raw, effort, ratings, dev_mode)
top_driver    = top_pos_feats[0] if top_pos_feats else "project size"
top_saver     = wi_results[0]["feature"] if wi_results else None

summary_html = executive_summary(
    effort, cal_months, team_size, eaf, loc, top_driver, top_saver, risk_label,
    ai_effort=ai_effort if ai_using else None,
    ai_cal=ai_cal, ai_team=ai_team, ai_pct=ai_pct, ai_short=ai_short,
)
st.markdown(f"""
<div class="exec-summary">
  <div class="exec-summary-label">Executive Summary</div>
  <div class="exec-summary-text">{summary_html}</div>
</div>""", unsafe_allow_html=True)

# ── SHAP tabs ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Explainability</div>", unsafe_allow_html=True)

if not SHAP_AVAILABLE:
    st.info("SHAP explainability is not available in this deployment environment. All other features work normally.", icon="ℹ️")
else:
    tab1, tab2, tab3 = st.tabs(["Feature Contributions", "FAHP vs SHAP", "Waterfall"])
    with tab1:
        _fig_bar = plot_shap_bar(sv_pm, effort)
        _ = st.pyplot(_fig_bar, width="stretch"); plt.close(_fig_bar)
        st.markdown("""<div class='insight-note' style='margin-top:0.4rem;'>
            <strong>Reading SHAP values:</strong> Each bar is the contribution of that feature
            relative to the model's <em>average prediction across all 63 training projects</em>
            — not relative to zero effort. A "Nominal" rating can still push effort up if the
            training average for that driver was High or Very High
            (COCOMO-81 teams were experienced 1980s defense engineers).
        </div>""", unsafe_allow_html=True)
    with tab2:
        _fig_fvs = plot_fahp_vs_shap(sv_pm)
        _ = st.pyplot(_fig_fvs, width="stretch"); plt.close(_fig_fvs)
    with tab3:
        _fig_wf = plot_waterfall(sv_pm, base_pm, effort)
        _ = st.pyplot(_fig_wf, width="stretch"); plt.close(_fig_wf)

# ── Insights ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
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
    st.markdown("**Effort drivers — pushing up**")
    for feat, val in top_pos:
        tag = _rating_tag(feat)
        tag_html = f" <span style='font-size:0.75rem;color:#999;font-weight:400;'>({tag})</span>" if tag else ""
        st.markdown(f"""<div class='insight-box insight-up'>
            <strong>{feat}</strong>{tag_html} &nbsp;·&nbsp; {FEAT_DESC.get(feat,feat)}<br>
            <span style='font-family:monospace;'>+{val:.1f} PM</span> above baseline
        </div>""", unsafe_allow_html=True)
with ic2:
    st.markdown("**Effort reducers — pushing down**")
    for feat, val in top_neg:
        tag = _rating_tag(feat)
        tag_html = f" <span style='font-size:0.75rem;color:#999;font-weight:400;'>({tag})</span>" if tag else ""
        st.markdown(f"""<div class='insight-box insight-down'>
            <strong>{feat}</strong>{tag_html} &nbsp;·&nbsp; {FEAT_DESC.get(feat,feat)}<br>
            <span style='font-family:monospace;'>{val:.1f} PM</span> below baseline
        </div>""", unsafe_allow_html=True)

# ── FAHP vs SHAP alignment ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FAHP Prior vs SHAP Posterior — Alignment</div>", unsafe_allow_html=True)
st.markdown("""<div class='explain-note'>
    <div class='explain-label'>Methodology note</div>
    <div class='explain-text'>
        <strong>FAHP weights</strong> capture data-driven feature importance — derived from Spearman
        correlations between each feature and project-level effort/LOC weights (an indirect proxy for
        expert pairwise comparisons). <strong>SHAP values</strong> reveal what the trained SVR model
        actually learned from 63 COCOMO-81 projects. Divergence between the two is itself a research
        finding: it shows where the FAHP weighting and empirical model behaviour agree or differ.
    </div>
</div>""", unsafe_allow_html=True)
f3 = np.argsort(fahp_w)[::-1][:3]
if sv_pm is not None:
    s3 = np.argsort(np.abs(sv_pm))[::-1][:3]
    agree = len(set(f3) & set(s3)) / 3 * 100
    colour = "#2E7D32" if agree >= 67 else ("#BF6000" if agree >= 33 else "#C62828")
    label_agree = "Strong agreement" if agree >= 67 else ("Partial" if agree >= 33 else "Disagreement")
    ac1, ac2 = st.columns([2.5, 1])
    with ac1:
        def _match_badge(i):
            if f3[i] == s3[i]:
                return "<span class='match-badge match-aligned'>Aligned</span>"
            elif f3[i] in s3:
                return "<span class='match-badge match-partial'>Partial</span>"
            else:
                return "<span class='match-badge match-divergent'>Divergent</span>"
        rows_html = "".join(
            f"<tr><td>#{i+1}</td><td>{FEATURES[f3[i]]}</td><td>{FEATURES[s3[i]]}</td><td>{_match_badge(i)}</td></tr>"
            for i in range(3)
        )
        st.markdown(f"""
        <table class='data-table'>
          <thead><tr>
            <th>Rank</th><th>FAHP Prior</th><th>SHAP Posterior</th><th>Match</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    with ac2:
        st.markdown(f"""<div class='metric-card' style='border-color:{colour};'>
            <div class='metric-label'>Alignment Score</div>
            <div class='metric-value' style='color:{colour};'>{agree:.0f}%</div>
            <div class='metric-unit'>{label_agree}</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("FAHP vs SHAP alignment requires SHAP explainability, which is not available in this environment.", icon="ℹ️")

# ── What-if ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What-If Analysis — Reduction Opportunities</div>", unsafe_allow_html=True)
if wi_results:
    wi_rows = "".join(
        f"<tr><td>{r['action']}</td>"
        f"<td style='font-family:monospace;font-size:0.77rem;color:#555;'>{r['feature']}</td>"
        f"<td>{r['from_']} &rarr; {r['to']}</td>"
        f"<td style='font-family:monospace;font-weight:600;color:#2E7D32;'>&#8722;{r['saving']:.1f}</td>"
        f"<td style='font-family:monospace;color:#2E7D32;'>{r['pct']:.1f}%</td></tr>"
        for r in wi_results
    )
    st.markdown(f"""
    <table class='data-table'>
      <thead><tr>
        <th>Action</th><th>Feature</th><th>Change</th><th>Saves (PM)</th><th>Reduction</th>
      </tr></thead>
      <tbody>{wi_rows}</tbody>
    </table>""", unsafe_allow_html=True)
    b = wi_results[0]
    st.markdown(f"""<div class='wi-callout'>
        <strong>Top recommendation:</strong> {b['action']}
        &nbsp;&middot;&nbsp; {b['feature']}: {b['from_']} &rarr; {b['to']}
        &nbsp;&middot;&nbsp; saves <strong>{b['saving']:.0f} PM ({b['pct']:.1f}%)</strong>
        <br><span style='font-size:0.78rem;opacity:0.8;'>
        Savings are relative to the COCOMO-81 training baseline (1980s defence/aerospace teams).
        A Nominal-rated driver may still appear as a cost driver if the training average was higher.
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.info("No improvement opportunities — already at optimal settings.")

# ── GenAI Impact Analysis ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>GenAI Impact Analysis</div>", unsafe_allow_html=True)
if ai_using:
    ga1, ga2 = st.columns([1.5, 1])
    with ga1:
        _fig_ai = plot_ai_comparison(effort, ai_effort, ai_effort_lo, ai_effort_hi, ai_short)
        _ = st.pyplot(_fig_ai, width="stretch"); plt.close(_fig_ai)
    with ga2:
        ai_pct_lo = (1 - ai_mult_hi) * 100
        ai_pct_hi = (1 - ai_mult_lo) * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>AI Saving</div>
            <div class='metric-value' style='color:#16A34A;font-size:1.85rem;'>{ai_pct:.0f}%</div>
            <div class='metric-unit'>effort reduction</div>
            <div class='metric-range'>{ai_pct_lo:.0f}–{ai_pct_hi:.0f}% study range</div>
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
            Productivity multipliers derived from published studies and industry surveys: <strong>{source_text}</strong>.
            COCOMO-81's TOOL driver caps out at 17% effort reduction; modern AI coding tools
            measurably exceed this — quantifying that gap is the novel contribution of this work.
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class='explain-note'>
        <div class='explain-label'>GenAI adjustment disabled</div>
        <div class='explain-text'>
            Select a GenAI adoption level in the sidebar to model productivity gains from
            Copilot, ChatGPT, Cursor, or similar AI coding tools.
            Multipliers are grounded in published research (Peng et al. 2023; McKinsey 2023;
            Kalliamvakou 2022).
        </div>
    </div>""", unsafe_allow_html=True)

# ── Validation Evidence ───────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Validation Evidence</div>", unsafe_allow_html=True)

ve1, ve2 = st.columns(2)

with ve1:
    st.markdown("**Ablation: Does FAHP actually help?**")
    if _abl_mmre and _fahp_imp:
        st.markdown(f"""<table class='data-table'>
          <thead><tr><th>Model</th><th>Nested-CV MMRE</th><th>PRED(25)</th></tr></thead>
          <tbody>
            <tr><td>SVR without FAHP (baseline)</td>
                <td style='font-family:monospace;color:#C62828;font-weight:600;'>{_abl_mmre:.4f}</td>
                <td style='color:#888;'>—</td></tr>
            <tr><td><strong>SVR with FAHP (this model)</strong></td>
                <td style='font-family:monospace;color:#2E7D32;font-weight:600;'>{_nested['mmre']:.4f}</td>
                <td style='font-family:monospace;color:#2E7D32;font-weight:600;'>{f"{_pred25*100:.0f}%" if _pred25 else "—"}</td></tr>
          </tbody>
        </table>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='wi-callout' style='margin-top:0.5rem;'>
            In matched nested validation, FAHP weighting changes MMRE by <strong>{_fahp_imp:.1f}%</strong>
            — from {_abl_mmre:.4f} to {_nested['mmre']:.4f}. This is a comparison,
            not proof that FAHP will improve every future dataset.
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
    <strong>Method:</strong> Feature weights are derived via Spearman correlation (a data-driven proxy for
    expert FAHP), and the solver is sklearn SVR rather than the LSSVM in Sehra et al. (2018).
    The weighted-kernel formulation (Eq.&nbsp;17) is faithfully implemented; the solver and weight-derivation
    are acknowledged simplifications.<br><br>
    <strong>GenAI adjustment:</strong> Post-hoc multipliers from external studies, not trained into the model.
    Applied to total effort; actual productivity gains in coding-only tasks may be higher.
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

1. **FAHP Feature Weighting** — Chang (1996) extent analysis builds project-level pairwise matrices from the effort/LOC distributions of all 63 training projects. Feature importance is derived via Spearman correlation. Weights embedded as S = diag(√θ) into the SVR kernel.

2. **Weighted-Kernel SVR** — sklearn SVR with RBF kernel, trained in log-space on COCOMO-81 (n=63). Kernel: K(θxₖ, θxₗ) = exp(−γ‖√θ(xₖ−xₗ)‖²).

3. **Nested validation** — Ten outer folds are used only for evaluation. Within each outer-training partition, five-fold inner CV selects hyperparameters and all scaling/FAHP fitting is repeated without seeing the outer-test rows. The deployment model is selected separately after evaluation.

4. **Empirical 80% PI** — 10th/90th percentile of pure outer-fold signed relative errors. No parametric assumption.

5. **SHAP** — KernelExplainer (background=50 representative projects ≤ 600 PM, nsamples=150, seed=42). Background excludes 13 defence mega-programs (up to 11,400 PM) to keep the baseline interpretable (~116 PM, near the geometric mean). SHAP values are in person-months; additivity verified at 0.00% error.

6. **GenAI Adjustment** — Post-hoc multiplier from Peng et al. 2023, McKinsey 2023, Kalliamvakou 2022. Not trained into the model.
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
    FAHP (Chang 1996) + Weighted-Kernel SVR (Sehra et al. 2018) &nbsp;&middot;&nbsp;
    Nested-CV MMRE&nbsp;{mmre_val:.4f} &nbsp;&middot;&nbsp; residual q10/q90&nbsp;[{PI_Q10*100:+.0f}%,&nbsp;{PI_Q90*100:+.0f}%]
  </div>
  <div class='footer-right'>v0.2-demo</div>
</div>
""", unsafe_allow_html=True)
