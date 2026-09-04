import streamlit as st
import numpy as np
import pickle, os

from app_metrics import model_metrics
from validation import prediction_interval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CSS ────────────────────────────────────────────────────────────────────────
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

section[data-testid="stSidebar"] {
    background: #0F0F0F !important;
    border-right: 1px solid #1E1E1E !important;
}
section[data-testid="stSidebar"] > div { background: #0F0F0F !important; }
section[data-testid="stSidebar"] * { color: #CCCCCC; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #BBBBBB !important; font-size: 0.78rem !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: #1C1C1C !important; border-color: #383838 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #DDDDDD !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] details summary p {
    color: #CCCCCC !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    border-bottom: 1px solid #2A2A2A !important;
    background: transparent !important;
}
.sb-section {
    font-size: 0.72rem; color: #888888; letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600; margin: 0.9rem 0 0.3rem 0;
}
.sb-val {
    font-size: 0.75rem; color: #666666; font-family: monospace;
    margin-top: -0.4rem; margin-bottom: 0.65rem;
}
.page-header {
    margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid #E2E2E2;
}
.page-title { font-size: 1.15rem; font-weight: 650; color: #111111; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
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
.metric-card {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 10px;
    padding: 1.2rem 1.4rem; height: 100%;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}
.metric-card:hover { border-color: #C8C8C8; box-shadow: 0 2px 14px rgba(0,0,0,0.07); }
.metric-label { font-size: 0.68rem; color: #666666; letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem; }
.metric-value { font-family: "JetBrains Mono", monospace; font-size: 2.2rem; font-weight: 600; color: #111111; line-height: 1; }
.metric-unit  { font-size: 0.78rem; color: #888888; margin-top: 0.28rem; }
.metric-range { font-family: "JetBrains Mono", monospace; font-size: 0.68rem; color: #888888; margin-top: 0.42rem; padding-top: 0.42rem; border-top: 1px solid #F0F0F0; }
.section-header {
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 700; color: #555555; margin: 2rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-header::after { content: ""; flex: 1; height: 1px; background: #EBEBEB; }
.explain-note {
    background: #FFFFFF; border: 1px solid #E2E2E2;
    border-left: 2px solid #AAAAAA; border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.9rem; margin: 0.5rem 0 0.9rem 0;
}
.explain-label { font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600; color: #AAAAAA; margin-bottom: 0.32rem; }
.explain-text  { font-size: 0.8rem; color: #444444; line-height: 1.65; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

DARK_CSS = """<style>
.stApp, section.main, .main { background: #111111 !important; }
.main .block-container { background: #111111 !important; }
.page-header { border-bottom-color: #2A2A2A !important; }
.page-title  { color: #EEEEEE !important; }
.page-subtitle { color: #888888 !important; }
.stat-pill       { background: #1E1E1E !important; border-color: #333 !important; color: #AAAAAA !important; }
.stat-pill-blue  { background: #0F1E33 !important; border-color: #1E3A5F !important; color: #5B9BD5 !important; }
.stat-pill-green { background: #0D2218 !important; border-color: #1A4A25 !important; color: #4CAF72 !important; }
.metric-card   { background: #1A1A1A !important; border-color: #2A2A2A !important; }
.metric-card:hover { border-color: #3A3A3A !important; }
.metric-value  { color: #EEEEEE !important; }
.metric-label  { color: #888888 !important; }
.metric-unit   { color: #666666 !important; }
.metric-range  { color: #666666 !important; border-top-color: #2A2A2A !important; }
.section-header { color: #777777 !important; }
.section-header::after { background: #2A2A2A !important; }
.explain-note  { background: #1A1A1A !important; border-color: #2A2A2A !important; }
.explain-label { color: #666 !important; }
.explain-text  { color: #CCCCCC !important; }
</style>"""

if st.session_state.get("dark_mode", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
_path = os.path.join(ROOT, "github_model.pkl")

@st.cache_data(show_spinner=False)
def load(mtime):
    with open(os.path.join(ROOT, "github_model.pkl"), "rb") as f:
        return pickle.load(f)

if not os.path.exists(_path):
    st.error("github_model.pkl not found. Run `python train_github_model.py`.")
    st.stop()

art         = load(os.path.getmtime(_path))
METRICS     = model_metrics("github")
NESTED      = METRICS["nested_cv"]
POPULARITY  = METRICS["popularity_holdout"]
FAHP        = METRICS["fahp_comparison"]
PROXY       = METRICS["effort_proxy"]
gh_model    = art["model"]
gh_scaler_X = art["scaler_X"]
gh_scaler_y = art["scaler_y"]
gh_kernel_w = art["kernel_w"]
gh_fahp_w   = art["fahp_weights"]
gh_feat     = art["feature_names"]
gh_pi_q10   = METRICS["prediction_interval"]["q10"]
gh_pi_q90   = METRICS["prediction_interval"]["q90"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0 0 0.8rem 0;border-bottom:1px solid #1E1E1E;margin-bottom:0.8rem;'>
      <div style='font-size:0.85rem;font-weight:600;color:#DDDDDD;letter-spacing:-0.01em;'>GitHub Effort Model</div>
      <div style='font-size:0.67rem;color:#555555;margin-top:0.1rem;font-family:monospace;letter-spacing:0.04em;'>FAHP &middot; SVR &middot; Real Projects</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>Team</div>", unsafe_allow_html=True)
    gh_contributors = st.slider("Team size", 5, 400, 50,
        help="Unique people who committed to the project", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_contributors} contributors</div>", unsafe_allow_html=True)

    gh_review_days = st.slider("PR review time", 0.0, 15.0, 2.0, 0.1,
        help="Median time from PR opened to merged", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_review_days:.1f} day avg review</div>", unsafe_allow_html=True)

    gh_merge_rate = st.slider("PR merge rate", 0.1, 1.0, 0.65, 0.01,
        help="Fraction of opened PRs that get merged", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_merge_rate:.0%} merge rate</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>Codebase</div>", unsafe_allow_html=True)
    gh_test_ratio = st.slider("Test file ratio", 0.0, 0.6, 0.20, 0.01,
        help="Fraction of repo files that are test/spec files", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_test_ratio:.0%} test files</div>", unsafe_allow_html=True)

    gh_lang_count = st.slider("Language count", 1, 20, 4,
        help="Number of programming languages in the repo", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_lang_count} language(s)</div>", unsafe_allow_html=True)

    gh_commit_freq = st.slider("Commit frequency", 0.5, 150.0, 10.0, 0.5,
        help="Average commits per week over the project window", label_visibility="collapsed")
    st.markdown(f"<div class='sb-val'>{gh_commit_freq:.0f} commits/week</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>AI Tooling</div>", unsafe_allow_html=True)
    gh_ai_tools = st.selectbox("AI tools", [0, 1, 2, 3], index=0,
        format_func=lambda x: {
            0: "None",
            1: "1 tool  (e.g. Copilot)",
            2: "2 tools  (Copilot + Claude)",
            3: "3 tools  (fully AI-first)",
        }[x],
        help="Number of distinct AI coding tools configured",
        label_visibility="collapsed")
    st.markdown(
        f"<div class='sb-val'>{'No AI tools adopted' if gh_ai_tools == 0 else f'{gh_ai_tools} AI tool(s) adopted'}</div>",
        unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #27272A;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.toggle("Dark mode", value=st.session_state.get("dark_mode", False), key="dark_mode")

# ── Predict ───────────────────────────────────────────────────────────────────
gh_contributors_x_freq = gh_contributors * gh_commit_freq
x_raw = np.array([[gh_contributors, gh_review_days, gh_merge_rate,
                   gh_test_ratio, gh_lang_count, gh_commit_freq,
                   gh_contributors_x_freq, gh_ai_tools]], dtype=float)
x_log = np.log(np.clip(x_raw, 1e-3, None))
x_sc  = gh_scaler_X.transform(x_log)
x_w   = x_sc * gh_kernel_w
p_sc  = gh_model.predict(x_w)
p_log = gh_scaler_y.inverse_transform(p_sc.reshape(-1, 1))[0, 0]
effort = max(float(np.expm1(p_log)), 1.0)
lo, hi = prediction_interval(effort, gh_pi_q10, gh_pi_q90, minimum=1.0)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div class="page-title">GitHub-Trained Effort Model</div>
  <div class="page-subtitle">FAHP + Weighted-Kernel SVR &nbsp;·&nbsp; Trained on {METRICS['n']} open-source GitHub projects</div>
  <div>
    <span class="stat-pill-blue">Nested-CV MMRE {NESTED['mmre']:.4f}</span>
    <span class="stat-pill-green">PRED(25) {NESTED['pred25']*100:.0f}%</span>
    <span class="stat-pill-blue">Popularity hold-out {POPULARITY['mmre']:.4f}</span>
    <span class="stat-pill">n={METRICS['n']} repos</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""<div class='explain-note'>
  <div class='explain-label'>What this model is</div>
  <div class='explain-text'>
    Same methodology as the COCOMO-81 model — FAHP feature weighting + weighted-kernel SVR —
    but trained on <strong>real GitHub open-source projects</strong> using signals your team
    already generates: PR review time, commit frequency, team size, test coverage.
    Effort is proxied from sampled commit author-days. Open-source volunteers differ from paid engineers,
    so this is a proof of concept. Company outcomes may be more representative, but their accuracy must be
    measured with held-out completed projects rather than assumed.
  </div>
</div>""", unsafe_allow_html=True)


# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Prediction</div>", unsafe_allow_html=True)
r1, r2, r3, r4, r5 = st.columns(5)
r1.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Predicted Effort</div>
    <div class='metric-value'>{effort:.0f}</div>
    <div class='metric-unit'>author-months</div>
    <div class='metric-range'>80% range: {lo:.0f}–{hi:.0f} AM</div>
</div>""", unsafe_allow_html=True)
r2.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Nested-CV MMRE</div>
    <div class='metric-value' style='color:#2563EB;font-size:1.8rem;'>{NESTED['mmre']:.4f}</div>
    <div class='metric-unit'>10 outer / 5 inner folds</div>
    <div class='metric-range'>PRED(25) = {NESTED['pred25']*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
r3.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Popularity Hold-out</div>
    <div class='metric-value' style='color:#15803d;font-size:1.8rem;'>{POPULARITY['mmre']:.4f}</div>
    <div class='metric-unit'>high-star train → low-star test</div>
    <div class='metric-range'>PRED(25) = {POPULARITY['pred25']*100:.0f}% · n_test={POPULARITY['test_n']}</div>
</div>""", unsafe_allow_html=True)
r4.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>FAHP Comparison</div>
    <div class='metric-value' style='color:#7c3aed;font-size:1.8rem;'>{FAHP['effect_pct']:+.1f}%</div>
    <div class='metric-unit'>lower MMRE in matched nested CV</div>
    <div class='metric-range'>Unweighted {FAHP['unweighted_mmre']:.4f} → weighted {FAHP['weighted_mmre']:.4f}</div>
</div>""", unsafe_allow_html=True)
r5.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Training Repos</div>
    <div class='metric-value' style='font-size:1.8rem;'>{METRICS['n']}</div>
    <div class='metric-unit'>open-source projects</div>
    <div class='metric-range'>GitHub API sample</div>
</div>""", unsafe_allow_html=True)

st.warning(
    f"The effort proxy samples at most {PROXY['commit_sample_limit']:,} commits per repository, "
    f"approximately {PROXY['approximate_cap_author_months']:.0f} author-months. Predictions above "
    "that cap are extrapolations and have not been validated."
)
st.info(
    "Popularity hold-out is not temporal validation: repository stars are not a time variable. "
    "Hyperparameters were selected only on the high-star training partition."
)

# ── Confidence band ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>80% Prediction Interval</div>", unsafe_allow_html=True)

fig_cb, ax_cb = plt.subplots(figsize=(7, 2.0))
fig_cb.patch.set_facecolor("white")
ax_cb.set_facecolor("white")
row_h    = 0.52
x_min_cb = lo * 0.58
x_max_cb = hi * 1.16
ax_cb.set_xlim(x_min_cb, x_max_cb)
ax_cb.barh([0], [hi - lo], left=[lo], height=row_h, color="#2563EB", alpha=0.10, edgecolor="none")
ax_cb.barh([0], [hi - lo], left=[lo], height=row_h, color="none", edgecolor="#2563EB", linewidth=1.1, alpha=0.6)
ax_cb.plot([effort], [0], marker="|", markersize=20, color="#2563EB", markeredgewidth=2.5, linewidth=0)
ax_cb.text(lo,     row_h * 0.65, f"{lo:.0f}",     ha="center", va="bottom", fontsize=7.5, color="#71717A", fontfamily="monospace")
ax_cb.text(hi,     row_h * 0.65, f"{hi:.0f}",     ha="center", va="bottom", fontsize=7.5, color="#71717A", fontfamily="monospace")
ax_cb.text(effort, row_h * 0.65, f"{effort:.0f}", ha="center", va="bottom", fontsize=8, color="#2563EB", fontfamily="monospace", fontweight="bold")
ax_cb.text(x_min_cb, 0, "Estimate", ha="right", va="center", fontsize=8, fontweight="bold", color="#2563EB")
ax_cb.set_yticks([])
ax_cb.set_xlabel("Author-months", fontsize=8, color="#71717A")
for sp in ["top", "right", "left"]: ax_cb.spines[sp].set_visible(False)
ax_cb.spines["bottom"].set_color("#E4E4E7")
ax_cb.tick_params(labelsize=7.5, colors="#71717A")
ax_cb.grid(axis="x", color="#F4F4F5", linewidth=0.7, zorder=0)
ax_cb.set_ylim(-row_h * 0.8, row_h * 1.4)
plt.tight_layout(pad=1.2)
st.pyplot(fig_cb, width="stretch")
plt.close(fig_cb)

# ── FAHP weights chart ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FAHP Feature Weights — What Drives Effort</div>", unsafe_allow_html=True)

order  = np.argsort(gh_fahp_w)
normed = gh_fahp_w / gh_fahp_w.max()
fig_fw, ax_fw = plt.subplots(figsize=(7, max(3.5, len(gh_fahp_w) * 0.50)))
fig_fw.patch.set_facecolor("white")
ax_fw.set_facecolor("#FAFAFA")
bars = ax_fw.barh(
    [gh_feat[i] for i in order],
    [gh_fahp_w[i] for i in order],
    color=[plt.cm.Blues(0.28 + 0.68 * normed[i]) for i in order],
    height=0.62, edgecolor="none",
)
for bar, i in zip(bars, order):
    ax_fw.text(
        gh_fahp_w[i] + gh_fahp_w.max() * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{gh_fahp_w[i]:.4f}", va="center", fontsize=8,
        fontfamily="monospace", color="#374151",
    )
ax_fw.set_xlabel("FAHP weight", fontsize=8, color="#71717A")
ax_fw.set_xlim(0, gh_fahp_w.max() * 1.20)
for sp in ["top", "right", "left"]: ax_fw.spines[sp].set_visible(False)
ax_fw.spines["bottom"].set_color("#E4E4E7")
ax_fw.tick_params(axis="y", labelsize=8.5, colors="#27272A")
ax_fw.tick_params(axis="x", labelsize=7.5, colors="#71717A")
ax_fw.grid(axis="x", color="#F0F0F0", linewidth=0.7)
plt.tight_layout()
st.pyplot(fig_fw, width="stretch")
plt.close(fig_fw)

st.markdown("""<div class='explain-note' style='margin-top:0.5rem;'>
  <div class='explain-label'>Interpretation</div>
  <div class='explain-text'>
    <strong>Contributors</strong> is the dominant predictor — more people means more coordination overhead and more effort.
    <strong>Commit Frequency</strong> captures active development intensity. <strong>Test Ratio</strong> reflects
    engineering maturity. On your company's data, the ranking would shift to reflect your specific team dynamics.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='section-header'>Validity — Transparency on Feature/Target Overlap</div>", unsafe_allow_html=True)
st.markdown(f"""<div class='explain-note' style='border-left-color:#BF6000;'>
  <div class='explain-label' style='color:#BF6000;'>Construct overlap — read before citing the MMRE</div>
  <div class='explain-text'>
    <strong>contributor_count</strong> has Spearman r = 0.70 with the effort proxy (author_months),
    and the interaction term r = 0.65, because all three derive from the same commit-activity log.
    This is <strong>construct overlap, not train/test leakage</strong>: author_months deduplicates to
    unique person-days and divides by 20, so it is strongly related to — but not a deterministic
    function of — contributor count. A 50-person team each committing once total has very different
    author_months than 50 people each active for months.
    <br><br>
    The low MMRE therefore reflects a genuine team-size → effort signal that cannot be cleanly
    separated from shared measurement source. Jira-tracked hours as an independent ground truth
    would remove the overlap and give a clean accuracy figure.
    <br><br>
    <em>An earlier version of this note reported r = 0.04; that was computed with Pearson, not the
    Spearman used everywhere else in the model. Corrected.</em>
  </div>
</div>""", unsafe_allow_html=True)
