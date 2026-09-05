"""
generate_pdf.py — generates Software_Effort_Estimator_Technical_Documentation.pdf
Run: python generate_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Software_Effort_Estimator_Technical_Documentation.pdf")
PROJECT_OUT = os.path.join(ROOT, "project_documentation.pdf")
HTML_OUT = os.path.join(ROOT, "Software_Effort_Estimator_Technical_Documentation.html")

with open(os.path.join(ROOT, "metrics.json"), encoding="utf-8") as metrics_file:
    REPORT = json.load(metrics_file)
C = REPORT["models"]["cocomo81"]
G = REPORT["models"]["github"]
C_CV, G_CV = C["nested_cv"], G["nested_cv"]
C_FAHP, G_FAHP = C["fahp_comparison"], G["fahp_comparison"]
NASA = C["external_nasa93"]
POPULARITY = G["popularity_holdout"]
C_PI = C["prediction_interval"]
PROXY = G["effort_proxy"]
EXPLAIN = C["explainability"]
EX_PROTOCOL = EXPLAIN["protocol"]
EX_PRIMARY = EXPLAIN["primary_alignment"]
EX_PRIMARY_REPEATS = EXPLAIN["primary_alignment_across_repeats"]
EX_WEIGHTED = EXPLAIN["weighted_alignment_descriptive"]
EX_WEIGHTED_REPEATS = EXPLAIN["weighted_alignment_across_repeats_descriptive"]
EX_UNWEIGHTED_GLOBAL_STABILITY = EXPLAIN["unweighted_global_ranking_stability"]
EX_UNWEIGHTED_LOCAL_STABILITY = EXPLAIN["unweighted_local_stability"]
EX_WEIGHTED_GLOBAL_STABILITY = EXPLAIN["weighted_global_ranking_stability"]
EX_WEIGHTED_LOCAL_STABILITY = EXPLAIN["weighted_local_stability"]
EX_BOOTSTRAP = EXPLAIN["primary_alignment_project_bootstrap"]
EX_PERMUTATION = EXPLAIN["feature_label_permutation_null"]
EX_NUMERICAL = EXPLAIN["numerical_sensitivity"]
EX_NUMERICAL_MAIN = next(
    row for row in EX_NUMERICAL["sample_size_results"]
    if row["nsamples"] == EX_PROTOCOL["nsamples"]
)
EX_PROXY_DIAGNOSTICS = EXPLAIN["weighting_proxy"]["full_data_diagnostics"]
EX_REPEAT_ACCURACY = EXPLAIN["repeated_accuracy_descriptive"]
EX_TOP_FEATURES = sorted(
    EXPLAIN["feature_table"], key=lambda row: row["unweighted_shap_rank"]
)[:5]

pdfmetrics.registerFont(TTFont("AppSans", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("AppSans-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("AppSans-Italic", r"C:\Windows\Fonts\ariali.ttf"))
pdfmetrics.registerFont(TTFont("AppSans-BoldItalic", r"C:\Windows\Fonts\arialbi.ttf"))
pdfmetrics.registerFontFamily(
    "AppSans", normal="AppSans", bold="AppSans-Bold",
    italic="AppSans-Italic", boldItalic="AppSans-BoldItalic",
)

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE      = colors.HexColor("#2563EB")
BLUE_LIGHT= colors.HexColor("#EFF6FF")
BLUE_BDR  = colors.HexColor("#BFDBFE")
GREEN     = colors.HexColor("#15803D")
GREEN_BG  = colors.HexColor("#F0FDF4")
PURPLE    = colors.HexColor("#7C3AED")
PURPLE_BG = colors.HexColor("#F5F3FF")
AMBER     = colors.HexColor("#BF6000")
AMBER_BG  = colors.HexColor("#FFFBEB")
RED       = colors.HexColor("#DC2626")
GREY_BG   = colors.HexColor("#F7F7F7")
GREY_BDR  = colors.HexColor("#E2E2E2")
DARK      = colors.HexColor("#111111")
MID       = colors.HexColor("#555555")
LIGHT     = colors.HexColor("#888888")
WHITE     = colors.white

W, H = A4
ML, MR, MT, MB = 2.2*cm, 2.2*cm, 2.5*cm, 2.5*cm

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE   = S("title_",   fontSize=28, leading=34, textColor=DARK,
            fontName="AppSans-Bold", spaceAfter=6, alignment=TA_LEFT)
SUBT    = S("subt",     fontSize=13, leading=18, textColor=LIGHT,
            fontName="AppSans",     spaceAfter=4, alignment=TA_LEFT)
H1      = S("h1",       fontSize=18, leading=24, textColor=DARK,
            fontName="AppSans-Bold", spaceBefore=20, spaceAfter=6, keepWithNext=True)
H2      = S("h2",       fontSize=13, leading=18, textColor=BLUE,
            fontName="AppSans-Bold", spaceBefore=14, spaceAfter=4)
H3      = S("h3",       fontSize=11, leading=15, textColor=MID,
            fontName="AppSans-Bold", spaceBefore=10, spaceAfter=3)
BODY    = S("body_",    fontSize=10, leading=16, textColor=DARK,
            fontName="AppSans",     spaceAfter=6,  alignment=TA_JUSTIFY)
BODYB   = S("bodyb",    fontSize=10, leading=16, textColor=DARK,
            fontName="AppSans-Bold",spaceAfter=6,  alignment=TA_JUSTIFY)
BULLET  = S("bullet_",  fontSize=10, leading=15, textColor=DARK,
            fontName="AppSans",     spaceAfter=3,
            leftIndent=16, firstLineIndent=-10)
MONO    = S("mono",     fontSize=9,  leading=13, textColor=BLUE,
            fontName="AppSans",       spaceAfter=4,
            backColor=BLUE_LIGHT,     borderPadding=(4,6,4,6))
CAPTION = S("caption",  fontSize=8.5,leading=12, textColor=LIGHT,
            fontName="AppSans",     spaceAfter=8,  alignment=TA_CENTER)
LABEL   = S("label_",   fontSize=7.5,leading=10, textColor=LIGHT,
            fontName="AppSans-Bold",spaceAfter=2,
            textTransform="uppercase", letterSpacing=0.8)
CALLOUT = S("callout",  fontSize=10, leading=15, textColor=colors.HexColor("#1A3A5C"),
            fontName="AppSans",     spaceAfter=0,  alignment=TA_JUSTIFY,
            backColor=BLUE_LIGHT,     borderPadding=(8,10,8,10))


def p(text, style=BODY):
    return Paragraph(text, style)

def sp(h=8):
    return Spacer(1, h)

def rule(color=GREY_BDR, thickness=0.5):
    line = HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=6)
    line.keepWithNext = True
    return line

def bullet(text, marker="•"):
    return Paragraph(f"{marker}&nbsp;&nbsp;{text}", BULLET)

def callout(text, bg=BLUE_LIGHT, border=BLUE):
    style = ParagraphStyle("co", parent=CALLOUT, backColor=bg,
                           borderPadding=(8,10,8,10))
    tbl = Table([[Paragraph(text, style)]], colWidths=[W - ML - MR - 0.4*cm])
    tbl.setStyle(TableStyle([
        ("BOX",    (0,0),(0,0), 1.5, border),
        ("LEFTPADDING",  (0,0),(0,0), 10),
        ("RIGHTPADDING", (0,0),(0,0), 10),
        ("TOPPADDING",   (0,0),(0,0), 8),
        ("BOTTOMPADDING",(0,0),(0,0), 8),
        ("BACKGROUND",   (0,0),(0,0), bg),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl

def section_tag(text, bg=BLUE_LIGHT, tc=BLUE):
    tbl = Table([[Paragraph(f"<font color='#{tc.hexval()[2:]}' size='7'><b>{text.upper()}</b></font>", LABEL)]],
                colWidths=[W - ML - MR - 0.4*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), bg),
        ("LEFTPADDING",  (0,0),(0,0), 8),
        ("RIGHTPADDING", (0,0),(0,0), 8),
        ("TOPPADDING",   (0,0),(0,0), 4),
        ("BOTTOMPADDING",(0,0),(0,0), 4),
    ]))
    return tbl

def formula_box(text):
    mono_style = ParagraphStyle("fb", fontSize=10, fontName="AppSans",
                                textColor=DARK, alignment=TA_CENTER,
                                backColor=GREY_BG, borderPadding=(8,12,8,12))
    tbl = Table([[Paragraph(text, mono_style)]], colWidths=[W - ML - MR - 0.4*cm])
    tbl.setStyle(TableStyle([
        ("BOX",          (0,0),(0,0), 0.8, GREY_BDR),
        ("BACKGROUND",   (0,0),(0,0), GREY_BG),
        ("LEFTPADDING",  (0,0),(0,0), 12),
        ("RIGHTPADDING", (0,0),(0,0), 12),
        ("TOPPADDING",   (0,0),(0,0), 8),
        ("BOTTOMPADDING",(0,0),(0,0), 8),
    ]))
    return tbl

def data_table(headers, rows, col_widths=None):
    col_widths = col_widths or [((W - ML - MR - 0.4*cm) / len(headers))] * len(headers)
    header_row = [Paragraph(f"<b>{h}</b>", ParagraphStyle("th", fontSize=8, fontName="AppSans-Bold",
                             textColor=MID, alignment=TA_LEFT)) for h in headers]
    data = [header_row]
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), ParagraphStyle("td", fontSize=9, fontName="AppSans",
                               textColor=DARK, leading=13, alignment=TA_LEFT)) for c in row])
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  GREY_BG),
        ("LINEBELOW",     (0,0), (-1,0),  0.8, GREY_BDR),
        ("LINEBELOW",     (0,1), (-1,-1), 0.4, colors.HexColor("#F0F0F0")),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GREY_BG]),
        ("BOX",           (0,0), (-1,-1), 0.6, GREY_BDR),
    ]))
    return tbl


# ── Page template (header + footer) ──────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(GREY_BDR)
    canvas.setLineWidth(0.5)
    canvas.line(ML, H - MT + 8, W - MR, H - MT + 8)
    if doc.page > 1:
        canvas.setFont("AppSans", 7.5)
        canvas.setFillColor(LIGHT)
        canvas.drawString(ML, H - MT + 12, "Software Effort Estimator — Technical Documentation")
        canvas.drawRightString(W - MR, H - MT + 12, f"Page {doc.page}")
    # Footer line
    canvas.line(ML, MB - 6, W - MR, MB - 6)
    canvas.setFont("AppSans", 7)
    canvas.setFillColor(LIGHT)
    canvas.drawString(ML, MB - 14, "FAHP-inspired proxy · Weighted RBF-SVR · OOF KernelSHAP · Mitacs Research")
    canvas.drawRightString(W - MR, MB - 14, "Confidential — Research Demo")
    canvas.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT
# ══════════════════════════════════════════════════════════════════════════════
story = []

# ── COVER PAGE ────────────────────────────────────────────────────────────────
story += [
    sp(60),
    p("Software Effort Estimator", TITLE),
    rule(BLUE, 2),
    sp(4),
    p("Technical Documentation", SUBT),
    sp(4),
    p("How the Model Works — From Raw Inputs to Explainable Predictions", SUBT),
    sp(40),
]

cover_table = Table([
    [Paragraph("<b>Methodology</b>", LABEL),
     Paragraph("Sehra et al. (2019)-inspired weighted RBF-SVR + leakage-safe OOF KernelSHAP", BODY)],
    [Paragraph("<b>Training Data</b>", LABEL),
     Paragraph(f"COCOMO-81 ({C['n']} projects) · GitHub open-source ({G['n']} repos)", BODY)],
    [Paragraph("<b>Validation</b>", LABEL),
     Paragraph("Nested 10-fold outer / 5-fold inner CV · popularity hold-out · NASA-93 external test", BODY)],
    [Paragraph("<b>Headline MMRE</b>", LABEL),
     Paragraph(f"{C_CV['mmre']:.4f} (COCOMO-81) · {G_CV['mmre']:.4f} (GitHub)", BODY)],
    [Paragraph("<b>Research Context</b>", LABEL),
     Paragraph("Mitacs Accelerate Internship — Industry partner model deliverable", BODY)],
], colWidths=[3.5*cm, W - ML - MR - 0.4*cm - 3.5*cm])
cover_table.setStyle(TableStyle([
    ("LINEBELOW",     (0,0),(-1,-2), 0.4, GREY_BDR),
    ("LEFTPADDING",   (0,0),(-1,-1), 0),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
]))
story += [cover_table, sp(40),
          p("Keshav · Mitacs Research Demo · 2026", CAPTION),
          PageBreak()]


# ── SECTION 1: INTRODUCTION ───────────────────────────────────────────────────
story += [
    p("1. What Is Software Effort Estimation?", H1), rule(BLUE_BDR),
    p("""Software effort estimation answers one question: <b>how many person-months will
    it take to build this project?</b> Getting this right matters enormously. Underestimate
    and the team burns out trying to meet an impossible deadline. Overestimate and
    resources sit idle while competitors ship faster."""),
    sp(),
    p("""For decades, companies relied on rules of thumb, gut feel, or simple spreadsheets.
    The COCOMO-81 model (Boehm, 1981) was a landmark — the first systematic, empirically
    grounded approach. It said: <i>if you tell me the size of the codebase and 15 project
    attributes, I can give you a reliable effort estimate.</i>"""),
    sp(),
    p("""This research prototype builds on that foundation in three ways:"""),
    bullet("<b>Training-derived weighting</b> — a data-derived FAHP-inspired proxy assigns feature weights inside each training fold. It is not an independent panel of expert judgements."),
    bullet("<b>Non-linear learning</b> — Support Vector Regression (SVR) with a weighted kernel learns patterns from historical data and is evaluated separately from the original multiplicative COCOMO formula."),
    bullet("<b>Explainability</b> — KernelSHAP attributes each model output to its inputs, in person-months, relative to a stated background dataset. These are model attributions, not causal effects."),
    sp(),
    callout("This is an adaptation, not a reproduction, of Sehra et al. (2019): their study used FAHP-weighted LSSVM, whereas this prototype uses a training-derived weighting proxy with sklearn SVR and adds an out-of-fold KernelSHAP analysis.", BLUE_LIGHT, BLUE),
    sp(12),

    p("1.1 The Core Metric: MMRE", H2),
    p("""Throughout this document you will see <b>MMRE</b> (Mean Magnitude of Relative Error).
    It is the standard accuracy measure for effort estimation:"""),
    formula_box("MMRE = (1/n) × Σ |actual_effort − predicted_effort| / actual_effort"),
    p("""An MMRE of 0.30 means predictions are, on average, 30% away from the true value.
    Lower is better. PRED(25) is the complementary metric: the fraction of predictions
    within 25% of the true value. Higher is better."""),
    sp(),
    data_table(
        ["MMRE", "What it means", "Context"],
        [
            ["< 0.10", "Excellent — within 10% on average", "Rarely achieved on real projects"],
            ["0.10 – 0.25", "Lower relative error in this sample", f"GitHub nested-CV result: {G_CV['mmre']:.4f}"],
            ["0.25 – 0.40", "Moderate relative error in this sample", f"COCOMO nested-CV result: {C_CV['mmre']:.4f}"],
            ["> 0.50", "Poor — coin-flip territory", "Typical spreadsheet estimation"],
        ],
        col_widths=[2.5*cm, 8*cm, 6*cm]
    ),
]


# ── SECTION 2: COCOMO-81 ──────────────────────────────────────────────────────
story += [
    p("2. The COCOMO-81 Foundation", H1), rule(BLUE_BDR),
    p("""COCOMO stands for COnstructive COst MOdel. Barry Boehm published it in 1981 after
    analysing 63 software projects at TRW Aerospace. The fundamental equation is:"""),
    formula_box("Effort (PM) = a × KLOC^b × EAF"),
    p("""Where:"""),
    bullet("<b>KLOC</b> = thousands of lines of code (project size)"),
    bullet("<b>a, b</b> = constants that depend on development mode (Organic / Semi-detached / Embedded)"),
    bullet("<b>EAF</b> = Effort Adjustment Factor — a product of 15 cost driver multipliers"),
    sp(),
    p("2.1 Development Modes", H2),
    data_table(
        ["Mode", "Description", "a", "b", "Typical project"],
        [
            ["Organic", "Small team, familiar domain, flexible requirements", "2.4", "1.05", "Internal tools, small web apps"],
            ["Semi-detached", "Mixed experience, moderate constraints", "3.0", "1.12", "Mid-size commercial software"],
            ["Embedded", "Hard real-time, safety-critical, tight hardware constraints", "3.6", "1.20", "Avionics, medical devices"],
        ],
        col_widths=[3.2*cm, 5.8*cm, 1.4*cm, 1.4*cm, 4.8*cm]
    ),
    sp(),
    p("2.2 The 15 Cost Drivers (EAF)", H2),
    p("""Each cost driver is rated on a 5-point scale: Very Low, Low, Nominal, High, Very High.
    Each rating maps to a multiplier. The product of all 15 multipliers is the EAF.
    EAF &lt; 1.0 means the project conditions are <b>favourable</b> (reducing effort);
    EAF &gt; 1.0 means they are <b>challenging</b> (increasing effort)."""),
    sp(),
    data_table(
        ["Category", "Driver", "What it measures", "VL → VH range"],
        [
            ["Product",  "RELY", "Consequence of failure (inconvenience → safety-critical)", "0.75 → 1.40"],
            ["Product",  "DATA", "Database and data complexity", "0.94 → 1.16"],
            ["Product",  "CPLX", "Algorithmic and architectural complexity", "0.70 → 1.30"],
            ["Computer", "TIME", "CPU execution-time constraint", "1.00 → 1.30"],
            ["Computer", "STOR", "Memory constraint", "1.00 → 1.21"],
            ["Computer", "VIRT", "Platform/VM volatility", "0.87 → 1.30"],
            ["Computer", "TURN", "Development turnaround time", "0.87 → 1.15"],
            ["Personnel","ACAP", "Analyst/architect capability", "1.46 → 0.71"],
            ["Personnel","AEXP", "Application domain experience", "1.29 → 0.82"],
            ["Personnel","PCAP", "Programmer capability", "1.42 → 0.70"],
            ["Personnel","VEXP", "Virtual machine experience", "1.21 → 0.87"],
            ["Personnel","LEXP", "Programming language experience", "1.14 → 0.95"],
            ["Project",  "MODP", "Modern programming practices", "1.24 → 0.82"],
            ["Project",  "TOOL", "Software tools quality", "1.24 → 0.83"],
            ["Project",  "SCED", "Schedule constraint", "1.23 → 1.10"],
        ],
        col_widths=[2.2*cm, 1.8*cm, 8.7*cm, 3.9*cm]
    ),
    sp(),
    callout("""<b>Example:</b> A 50 KLOC Organic project with all Nominal ratings gives:<br/>
    Effort = 2.4 × 50^1.05 × 1.0 = 2.4 × 60.8 = <b>~146 person-months</b><br/>
    If the team has Very High programmer capability (PCAP multiplier = 0.70) and
    High complexity (CPLX = 1.15):<br/>
    EAF = 0.70 × 1.15 = 0.805 → Effort = 146 × 0.805 = <b>~118 person-months</b>""",
    BLUE_LIGHT, BLUE),
]


# ── SECTION 3: FAHP ──────────────────────────────────────────────────────────
story += [
    p("3. Fuzzy Analytic Hierarchy Process (FAHP)", H1), rule(BLUE_BDR),
    p("""COCOMO uses calibrated effort multipliers whose ranges differ by cost driver; it
    does not provide a separate, learned ranking of feature importance. Sehra et al. (2019)
    addressed feature ranking with FAHP before integrating the resulting weights into an
    LSSVM. The present sklearn-SVR implementation uses a <b>training-derived FAHP-inspired proxy</b>
    and must not be described as a faithful reproduction of that method."""),
    sp(),

    p("3.1 What Is the Analytic Hierarchy Process?", H2),
    p("""AHP (Saaty, 1980) is a decision-making framework that assigns weights to criteria
    by asking: <i>how much more important is criterion A than criterion B?</i> You build
    a matrix of pairwise comparisons and extract weights mathematically."""),
    sp(),
    p("""The problem with standard AHP is that human judgements are imprecise. Saying
    "CPLX is 3× more important than LEXP" is hard to justify exactly.
    <b>Fuzzy AHP</b> replaces exact numbers with triangular fuzzy numbers (TFNs) that
    capture this uncertainty."""),
    p("""That is the motivation for expert-elicited FAHP. No independent expert panel was
    available for this prototype, so its current weights are supervised, data-derived proxies.
    A paper should keep that distinction explicit."""),
    sp(),

    p("3.2 Triangular Fuzzy Numbers", H2),
    p("""A TFN is written as (l, m, u): a lower bound, a most likely value, and an upper
    bound. Instead of saying "A is exactly 3× more important than B", we say
    "A is between 1× and 5× more important, most likely 3×" — written as (1, 3, 5)."""),
    formula_box("TFN = (l, m, u)   where   l ≤ m ≤ u"),
    p("""The TFN scale used in this model (Buckley 1985 / Chang 1996):"""),
    data_table(
        ["Linguistic judgement", "TFN"],
        [
            ["Equally important",         "(1, 1, 1)"],
            ["Weakly more important",      "(1, 2, 3)"],
            ["Moderately more important",  "(1, 3, 5)"],
            ["Strongly more important",    "(3, 5, 7)"],
            ["Very strongly more important","(5, 7, 9)"],
            ["Absolutely more important",  "(7, 9, 9)"],
        ],
        col_widths=[9*cm, 7.6*cm]
    ),
    sp(),

    p("3.3 Training-Derived FAHP-Inspired Proxy Used Here", H2),
    p("""The implemented COCOMO weighting procedure operates only on the relevant training
    rows and has three stages:"""),
    sp(),
    bullet("<b>Step 1 — Project comparisons:</b> Build separate fuzzy pairwise matrices across training projects from their effort and LOC ratios; outer-test outcomes never enter these matrices.", "1."),
    sp(4),
    bullet("<b>Step 2 — Project weights:</b> Apply Chang's synthetic-extent and degree-of-possibility calculations to obtain effort-based and LOC-based project weights, then average and normalise them.", "2."),
    sp(4),
    formula_box("S_i = (Σ_j l_ij / Σ_i Σ_j u_ij ,  Σ_j m_ij / Σ_i Σ_j m_ij ,  Σ_j u_ij / Σ_i Σ_j l_ij)"),
    sp(4),
    bullet("<b>Step 3 — Feature proxy weights:</b> Take the absolute Spearman association between each feature and the combined project weights, clip degenerate values, and normalise across features.", "3."),
    sp(4),
    formula_box("theta_i = normalise(|Spearman(feature_i, combined training-project weight)|)"),
    sp(),
    callout("""<b>Scope:</b> These weights are useful as a supervised ranking and kernel-scaling
    mechanism, but they are not independent expert priors. A faithful replication-and-extension
    study should obtain the original expert judgements or elicit a new documented panel.""", AMBER_BG, AMBER),
    sp(),

    p("3.4 How the Proxy Enters the Kernel", H2),
    p("""The normalised proxy vector theta is applied as an element-wise square-root
    transformation before the RBF kernel. This makes theta the coefficient of each feature's
    squared-distance contribution. Predictive benefit is assessed separately with a matched,
    independently tuned weighted-versus-unweighted nested-validation comparison."""),
]


# ── SECTION 4: SVR ───────────────────────────────────────────────────────────
story += [
    p("4. Weighted-Kernel Support Vector Regression (SVR)", H1), rule(BLUE_BDR),
    p("""Once the training-derived proxy produces feature weights, an SVR learns non-linear
    patterns from the 63 COCOMO-81 projects. This is sklearn epsilon-SVR, not the LSSVM solver
    used by Sehra et al. (2019)."""),
    sp(),

    p("4.1 What Is SVR?", H2),
    p("""Support Vector Regression learns a function that predicts continuous outputs
    (like person-months). Unlike linear regression, it uses a <b>kernel function</b>
    to measure similarity between projects in a high-dimensional feature space,
    allowing it to capture non-linear relationships."""),
    sp(),
    p("""The key idea: instead of minimising the sum of squared errors (like OLS regression),
    SVR tries to find a function where most predictions fall within an ε-tube around the
    true values. Points outside the tube are penalised by a cost parameter C."""),
    formula_box("SVR finds f(x) such that |y_i - f(x_i)| ≤ ε for most training points"),
    sp(),

    p("4.2 The RBF Kernel", H2),
    p("""This model uses the Radial Basis Function (RBF) kernel, which measures
    how "close" two projects are in feature space:"""),
    formula_box("K(x_i, x_j) = exp(-γ × ||x_i - x_j||²)"),
    p("""Two projects that are very similar (small distance) have K ≈ 1.
    Two projects that are very different have K ≈ 0. The model learns that
    similar projects should have similar effort estimates.
    γ controls how quickly similarity falls off with distance — a hyperparameter
    tuned by cross-validation."""),
    sp(),

    p("4.3 Training-Derived Weighting in the RBF Kernel", H2),
    p("""Instead of measuring distance equally across all features, the implementation
    rescales the feature space with the training-derived proxy weights:"""),
    formula_box("K_theta(x_i, x_j) = exp(-gamma × ||sqrt(theta) · (x_i - x_j)||²)"),
    p("""Where theta is the vector of normalised proxy weights and sqrt(theta) is applied element-wise.
    This is equivalent to:"""),
    formula_box("x_weighted = x × sqrt(theta)"),
    p("""A larger theta therefore gives that feature a larger coefficient in squared RBF
    distance. This changes the model's geometry; it does not establish that the feature is a
    causal effort driver."""),
    sp(),

    p("4.4 The Full Training Pipeline", H2),
    data_table(
        ["Step", "What happens", "Why"],
        [
            ["1. Log transform", "x → log(x),  y → log(1 + y)", "Effort is right-skewed; log space makes it more symmetric"],
            ["2. Scale", "MinMaxScaler to [0, 1]", "SVR is sensitive to feature scale differences"],
            ["3. Proxy weight", "x_weighted = x_scaled × sqrt(theta)", "Rescales each feature's contribution to RBF distance"],
            ["4. SVR fit", "SVR(RBF, C, γ, ε).fit(X_weighted, y_scaled)", "Learn non-linear effort surface"],
            ["5. Predict", "Predict → inverse-scale → expm1", "Convert back from log-space to person-months"],
        ],
        col_widths=[2.5*cm, 7*cm, 7.1*cm]
    ),
    sp(),

    p("4.5 Hyperparameter Tuning", H2),
    p("""Three SVR hyperparameters are tuned by exhaustive grid search, using
    five-fold inner-CV MMRE as the objective inside each outer-training partition:"""),
    bullet("<b>C</b> (cost): penalty for predictions outside the ε-tube. Higher C = tighter fit, risk of overfit."),
    bullet("<b>γ</b> (gamma): how quickly the RBF kernel falls off. Higher γ = more local decisions."),
    bullet("<b>ε</b> (epsilon): the width of the no-penalty tube. Larger ε = more tolerance."),
    sp(),
    callout("""<b>Why nested validation?</b> A single resampling loop cannot both choose
    hyperparameters and provide an untouched performance estimate. Ten outer folds are used
    only for evaluation; within each outer-training partition, five inner folds select C,
    gamma and epsilon. Scaling and training-derived proxy weights are also fitted only on the relevant training
    rows. Deployment parameters are selected separately after evaluation.""", AMBER_BG, AMBER),
]


# ── SECTION 5: SHAP ──────────────────────────────────────────────────────────
story += [
    p("5. Out-of-Fold KernelSHAP Analysis", H1), rule(BLUE_BDR),
    p("""SHAP (SHapley Additive exPlanations; Lundberg and Lee, 2017) decomposes a
    <b>model output</b> relative to an expected output defined by a background dataset.
    Because the complete prediction function returns person-months, these attributions are
    also expressed in person-months. They describe model behaviour, not real-world causality."""),
    formula_box("model prediction = expected model output over background + sum of SHAP attributions"),
    sp(),

    p("5.1 Leakage-Safe Repeated Protocol", H2),
    p(f"""The research analysis uses <b>{EX_PROTOCOL['repeats']} repeats of
    {EX_PROTOCOL['outer_splits']}-fold outer validation</b>, with
    {EX_PROTOCOL['inner_splits']}-fold inner selection. In every outer fold, preprocessing,
    the training-derived weighting proxy, and hyperparameters are fitted without the outer-test
    projects. KernelSHAP then explains only those outer-test predictions, so every project is
    explained once per repeat."""),
    bullet(f"<b>Explainer:</b> {EX_PROTOCOL['explainer']}"),
    bullet(f"<b>Background:</b> {EX_PROTOCOL['background_size']}; {EX_PROTOCOL['background_source']}"),
    bullet(f"<b>Sampling:</b> nsamples={EX_PROTOCOL['nsamples']}, l1_reg={EX_PROTOCOL['l1_reg']}; all 17 features enter the weighted least-squares solution"),
    bullet(f"<b>Additivity check:</b> maximum stored error = {EXPLAIN['max_additivity_error_person_months']:.6f} person-months"),
    sp(),
    callout("""<b>Primary comparison:</b> the training-derived FAHP-inspired proxy is compared with
    normalised mean absolute out-of-fold SHAP reliance from an <i>unweighted</i> SVR. This is
    the cleaner comparison because the proxy weights were not injected into that model.
    Agreement with weighted-model SHAP is reported separately and is descriptive: some of it
    may be induced by the weighted kernel itself.""", BLUE_LIGHT, BLUE),
    sp(),

    p("5.2 Agreement Results", H2),
    data_table(
        ["Comparison", "Spearman rho", "Kendall tau-b", "JS similarity", "Top-3 Jaccard / Kuncheva"],
        [
            ["Primary pooled: proxy vs unweighted OOF SHAP",
             f"{EX_PRIMARY['spearman_rho']:.6f}", f"{EX_PRIMARY['kendall_tau_b']:.6f}",
             f"{EX_PRIMARY['jensen_shannon_similarity']:.6f}",
             f"{EX_PRIMARY['top_k_jaccard']:.6f} / {EX_PRIMARY['top_k_kuncheva']:.6f} ({EX_PRIMARY['top_k_intersection']}/3 shared)"],
            ["Primary repeat median [min, max]",
             f"{EX_PRIMARY_REPEATS['spearman_rho']['median']:.6f}<br/>[{EX_PRIMARY_REPEATS['spearman_rho']['minimum']:.6f}, {EX_PRIMARY_REPEATS['spearman_rho']['maximum']:.6f}]",
             f"{EX_PRIMARY_REPEATS['kendall_tau_b']['median']:.6f}<br/>[{EX_PRIMARY_REPEATS['kendall_tau_b']['minimum']:.6f}, {EX_PRIMARY_REPEATS['kendall_tau_b']['maximum']:.6f}]",
             f"{EX_PRIMARY_REPEATS['jensen_shannon_similarity']['median']:.6f}<br/>[{EX_PRIMARY_REPEATS['jensen_shannon_similarity']['minimum']:.6f}, {EX_PRIMARY_REPEATS['jensen_shannon_similarity']['maximum']:.6f}]",
             f"{EX_PRIMARY_REPEATS['top_k_jaccard']['median']:.6f} / {EX_PRIMARY_REPEATS['top_k_kuncheva']['median']:.6f}"],
            ["Weighted pooled (descriptive; partly induced)",
             f"{EX_WEIGHTED['spearman_rho']:.6f}", f"{EX_WEIGHTED['kendall_tau_b']:.6f}",
             f"{EX_WEIGHTED['jensen_shannon_similarity']:.6f}",
             f"{EX_WEIGHTED['top_k_jaccard']:.6f} / {EX_WEIGHTED['top_k_kuncheva']:.6f} ({EX_WEIGHTED['top_k_intersection']}/3 shared)"],
            ["Weighted repeat median [min, max]",
             f"{EX_WEIGHTED_REPEATS['spearman_rho']['median']:.6f}<br/>[{EX_WEIGHTED_REPEATS['spearman_rho']['minimum']:.6f}, {EX_WEIGHTED_REPEATS['spearman_rho']['maximum']:.6f}]",
             f"{EX_WEIGHTED_REPEATS['kendall_tau_b']['median']:.6f}<br/>[{EX_WEIGHTED_REPEATS['kendall_tau_b']['minimum']:.6f}, {EX_WEIGHTED_REPEATS['kendall_tau_b']['maximum']:.6f}]",
             f"{EX_WEIGHTED_REPEATS['jensen_shannon_similarity']['median']:.6f}<br/>[{EX_WEIGHTED_REPEATS['jensen_shannon_similarity']['minimum']:.6f}, {EX_WEIGHTED_REPEATS['jensen_shannon_similarity']['maximum']:.6f}]",
             f"{EX_WEIGHTED_REPEATS['top_k_jaccard']['median']:.6f} / {EX_WEIGHTED_REPEATS['top_k_kuncheva']['median']:.6f}"],
        ],
        col_widths=[5.3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.8*cm]
    ),
    sp(),
    p("""The primary pooled rank association is moderate, while only one of the three
    highest-ranked features is shared. The larger weighted-model association must not be read
    as independent confirmation because the proxy weights alter that model's geometry."""),
    p(f"""A {EX_BOOTSTRAP['resamples']:,}-resample project-level percentile procedure gives
    a conditional {EX_BOOTSTRAP['coverage']:.0%} interval for primary Spearman rho of
    [{EX_BOOTSTRAP['metrics']['spearman_rho']['lower']:.6f},
    {EX_BOOTSTRAP['metrics']['spearman_rho']['upper']:.6f}] (resampling median
    {EX_BOOTSTRAP['metrics']['spearman_rho']['median']:.6f}). Projects are resampled after
    averaging their repeated OOF explanations. It does not refit models, regenerate splits,
    retune hyperparameters, or rerun the explainer, so it is not a full uncertainty interval."""),
    sp(),

    p("5.3 Repeat Stability", H2),
    data_table(
        ["Primary unweighted evidence", "Pairs", "Statistic", "Median [min, max]"],
        [
            ["Joint repeat ranking", f"{EX_UNWEIGHTED_GLOBAL_STABILITY['pair_count']}", "Spearman rho",
             f"{EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['median']:.6f} [{EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['minimum']:.6f}, {EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['maximum']:.6f}]"],
            ["Joint repeat ranking", f"{EX_UNWEIGHTED_GLOBAL_STABILITY['pair_count']}", "Kendall tau-b",
             f"{EX_UNWEIGHTED_GLOBAL_STABILITY['kendall_tau_b']['median']:.6f} [{EX_UNWEIGHTED_GLOBAL_STABILITY['kendall_tau_b']['minimum']:.6f}, {EX_UNWEIGHTED_GLOBAL_STABILITY['kendall_tau_b']['maximum']:.6f}]"],
            ["Joint repeat ranking", f"{EX_UNWEIGHTED_GLOBAL_STABILITY['pair_count']}", "Top-3 Jaccard",
             f"{EX_UNWEIGHTED_GLOBAL_STABILITY['top_k_jaccard']['median']:.6f} [{EX_UNWEIGHTED_GLOBAL_STABILITY['top_k_jaccard']['minimum']:.6f}, {EX_UNWEIGHTED_GLOBAL_STABILITY['top_k_jaccard']['maximum']:.6f}]"],
            ["Joint local attribution", f"{EX_UNWEIGHTED_LOCAL_STABILITY['comparison_count']}", "Cosine similarity",
             f"{EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['median']:.6f} [{EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['minimum']:.6f}, {EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['maximum']:.6f}]"],
            ["Joint local attribution", f"{EX_UNWEIGHTED_LOCAL_STABILITY['comparison_count']}", "Sign agreement",
             f"{EX_UNWEIGHTED_LOCAL_STABILITY['sign_agreement']['median']:.6f} [{EX_UNWEIGHTED_LOCAL_STABILITY['sign_agreement']['minimum']:.6f}, {EX_UNWEIGHTED_LOCAL_STABILITY['sign_agreement']['maximum']:.6f}]"],
            ["Joint local attribution", f"{EX_UNWEIGHTED_LOCAL_STABILITY['comparison_count']}", "Top-3 Jaccard",
             f"{EX_UNWEIGHTED_LOCAL_STABILITY['top_k_jaccard']['median']:.6f} [{EX_UNWEIGHTED_LOCAL_STABILITY['top_k_jaccard']['minimum']:.6f}, {EX_UNWEIGHTED_LOCAL_STABILITY['top_k_jaccard']['maximum']:.6f}]"],
        ],
        col_widths=[4.3*cm, 2.0*cm, 3.4*cm, 6.9*cm]
    ),
    p(f"""The split, fitted model, outer-training background, and KernelSHAP seed vary together,
    so these are joint resampling stability measurements rather than isolated model or explainer
    sensitivity. For the descriptive weighted model, joint repeat stability has median Spearman
    rho {EX_WEIGHTED_GLOBAL_STABILITY['spearman_rho']['median']:.6f}
    [{EX_WEIGHTED_GLOBAL_STABILITY['spearman_rho']['minimum']:.6f},
    {EX_WEIGHTED_GLOBAL_STABILITY['spearman_rho']['maximum']:.6f}], and local cosine similarity
    has median {EX_WEIGHTED_LOCAL_STABILITY['cosine_similarity']['median']:.6f}
    [{EX_WEIGHTED_LOCAL_STABILITY['cosine_similarity']['minimum']:.6f},
    {EX_WEIGHTED_LOCAL_STABILITY['cosine_similarity']['maximum']:.6f}]. The low minima show
    why a single explanation should not be presented as universally stable."""),
    p(f"""A separate fixed-model numerical audit holds the model, background, and 12 rows fixed.
    At nsamples={EX_PROTOCOL['nsamples']} versus the {EX_NUMERICAL['reference_nsamples']}-sample
    reference, global Spearman rho is
    {EX_NUMERICAL_MAIN['global_agreement_to_1024_reference']['spearman_rho']:.6f},
    Jensen-Shannon similarity is
    {EX_NUMERICAL_MAIN['global_agreement_to_1024_reference']['jensen_shannon_similarity']:.6f},
    the top three are identical, and median local cosine similarity is
    {EX_NUMERICAL_MAIN['local_cosine_to_1024_reference']['median']:.6f}. Seed-to-seed median
    global rho is {EX_NUMERICAL['seed_global_ranking_stability']['spearman_rho']['median']:.6f}.
    With l1_reg=0, every audited local explanation retains all 17 features."""),
    p(f"""In a {EX_PERMUTATION['permutations']:,}-permutation feature-label benchmark,
    primary Spearman agreement has empirical p={EX_PERMUTATION['empirical_p_values']['spearman_rho']['empirical_p']:.6f}
    and Jensen-Shannon similarity p={EX_PERMUTATION['empirical_p_values']['jensen_shannon_similarity']['empirical_p']:.6f},
    while top-3 Jaccard overlap is not unusual (p={EX_PERMUTATION['empirical_p_values']['top_k_jaccard']['empirical_p']:.6f}).
    This is a narrow descriptive relabelling benchmark, not a population hypothesis test."""),
    sp(),

    p("5.4 Highest Unweighted OOF SHAP Reliance", H2),
    data_table(
        ["Feature", "Proxy weight (rank)", "Unweighted OOF SHAP (rank)", "Weighted OOF SHAP (rank)"],
        [
            [row["feature"],
             f"{row['mean_weighting_proxy']:.6f} (#{row['proxy_rank']})",
             f"{row['unweighted_oof_shap_reliance']:.6f} (#{row['unweighted_shap_rank']})",
             f"{row['weighted_oof_shap_reliance']:.6f} (#{row['weighted_shap_rank']})"]
            for row in EX_TOP_FEATURES
        ],
        col_widths=[3.5*cm, 4.1*cm, 4.5*cm, 4.5*cm]
    ),
    p("""OOF SHAP reliance is normalised mean absolute attribution. It is not a causal
    effect, and correlated COCOMO drivers may redistribute attribution. LOC ranks first in
    both the proxy and primary SHAP result; RELY and TIME rank second and third by unweighted
    OOF SHAP even though their proxy ranks are lower."""),
    sp(),

    p("5.5 Relation to Prior Explainability Work", H2),
    p("""Adding SHAP to software-effort estimation is not by itself novel. Najm, Zakrani,
    and Marzak (2023) reported model-agnostic explanations for SVR-RBF effort models, and
    Jayadi, Prasetya, and Widiyaningtyas (2025) applied SHAP to a simpler effort model.
    This pilot instead evaluates a specific FAHP-proxy/SHAP relationship with repeated,
    leakage-safe out-of-fold explanations and an unweighted primary comparator. It does not
    establish a first-in-field claim."""),
    callout(f"""<b>Interactive-app distinction:</b> the on-screen single-project explanation
    uses a fixed, outcome-conditioned 50-project background (projects with effort at or below
    600 PM), nsamples={EX_PROTOCOL['nsamples']}, and l1_reg=0. Its base value is the expected
    model output over those selected rows, not a representative population mean. The research
    results above instead use all outer-training rows as each fold's background.""", AMBER_BG, AMBER),
]


# ── SECTION 6: GENAI ──────────────────────────────────────────────────────────
story += [
    p("6. Post-hoc GenAI Sensitivity Scenario", H1), rule(BLUE_BDR),
    p("""COCOMO-81 was published in 1981. Its TOOL cost driver captures the quality
    of development tools; the multiplier at Very High is 0.83. Modern AI coding tools
    are not represented in the historical training data."""),
    sp(),

    p("6.1 External Sources Used for Illustrative Bounds", H2),
    data_table(
        ["Source", "Reported setting", "Reported association", "Use here"],
        [
            ["Peng et al. (2023) — GitHub", "Controlled coding task", "Faster task completion with Copilot; full-project effort was not measured", "Illustrative bound only"],
            ["McKinsey Global Institute (2023)", "Industry report", "A range of potential developer-productivity effects", "Illustrative bound only"],
            ["Kalliamvakou (2022) — GitHub blog", "Observed Copilot cohort", "Faster pull-request merging and self-reported experience", "Illustrative bound only"],
            ["COCOMO-81 TOOL driver", "Historical projects", "Very High TOOL multiplier is 0.83", "Already represented in model input"],
        ],
        col_widths=[4.5*cm, 2.5*cm, 7*cm, 2.6*cm]
    ),
    sp(),
    callout("""<b>Research question:</b> Sehra et al. (2019) did not evaluate modern AI-tool
    signals. The multipliers below are external, post-hoc scenarios rather than learned effects.
    Estimating any AI-associated change from company projects with recorded effort would be a
    future empirical contribution; this prototype does not claim that result yet.""", PURPLE_BG, PURPLE),
    sp(),

    p("6.2 How the Adjustment Works", H2),
    p("""The GenAI adjustment is a <b>post-hoc multiplier</b> applied to the SVR prediction.
    It is not baked into the model — it is applied on top, transparently:"""),
    formula_box("AI_effort = baseline_effort × AI_multiplier"),
    data_table(
        ["AI level", "Multiplier (central)", "Range", "Source"],
        [
            ["None — no AI tools",            "1.00", "—",          "—"],
            ["Some — Copilot / ChatGPT",       "0.78", "0.75–0.82", "Peng et al. 2023, McKinsey 2023"],
            ["Heavy — Cursor / AI-first team", "0.62", "0.55–0.70", "McKinsey 2023, Kalliamvakou 2022"],
        ],
        col_widths=[6*cm, 3.5*cm, 3*cm, 4.1*cm]
    ),
    p("""The range is shown as a scenario range in the UI, reflecting variation across the cited studies.
    Under the illustrative "Heavy AI" central multiplier, a 200 PM model output becomes
    200 × 0.62 = <b>124 PM</b>, a modelled difference of 76 PM (38%). Applying the
    scenario bounds gives 110–140 PM. These are arithmetic sensitivity results, not a
    causal forecast or a confidence interval."""),
    sp(),
    p("""<b>Important caveat:</b> These multipliers come from task-level studies and enterprise
    surveys, not from effort estimation research. Applying them to full-project effort
    estimates is an unvalidated extrapolation. The actual effect depends on the codebase,
    team, adoption, and task mix and must be measured independently."""),
]


# ── SECTION 7: GITHUB MODEL ───────────────────────────────────────────────────
story += [
    p("7. The GitHub-Trained Model", H1), rule(BLUE_BDR),
    p("""The COCOMO-81 model was trained on historical defense software. As an exploratory
    modern-data exercise, 91 public GitHub repositories were used to train an RBF-SVR with
    supervised correlation-based feature weighting. This is not the same FAHP-LSSVM method as
    Sehra et al. (2019), and it does not establish industry generalisation."""),
    sp(),

    p("7.1 Features", H2),
    data_table(
        ["Feature", "What it measures", "Possible association (not causal)"],
        [
            ["Contributors",       "Unique people who committed", "Team activity and the effort proxy share commit-derived inputs"],
            ["PR Review Time",     "Avg days from PR open to merge", "May reflect workflow latency or repository practice"],
            ["PR Merge Rate",      "Fraction of PRs merged", "May reflect review and contribution policy"],
            ["Test File Ratio",    "Fraction of files that are tests", "Repository-structure signal; not measured test quality"],
            ["Language Count",     "Number of programming languages", "Repository heterogeneity signal"],
            ["Commit Frequency",   "Commits per week", "Development-activity signal"],
            ["Contributors × Freq","Interaction term", "Combined team-activity scale"],
            ["AI Tool Adoption",   "Number of detectable AI configs", "Configuration presence; adoption intensity is unknown"],
        ],
        col_widths=[3.5*cm, 5.5*cm, 7.6*cm]
    ),
    sp(),

    p("7.2 The Effort Proxy", H2),
    p("""GitHub doesn't have Jira tickets with actual hours. Instead, effort is approximated as:"""),
    formula_box("author_months = unique (author, date) pairs across all commits / 20"),
    p("""A contributor who commits 10 times in one day still counts as 1 author-day.
    Dividing by 20 converts working days to months (approx. 20 working days/month).
    This is an imperfect proxy — it measures developer presence, not hours of productive
    work — and it has not been validated against recorded person-hours."""),
    callout(f"""<b>Sampling ceiling:</b> collection is capped at {PROXY['commit_sample_limit']:,}
    commits per repository, approximately {PROXY['approximate_cap_author_months']:.0f}
    author-months under this proxy. Predictions above that cap are extrapolations and have
    not been validated.""", AMBER_BG, AMBER),
    sp(),

    p("7.3 Transparency: Feature/Target Overlap", H2),
    p("""Since contributor_count, the interaction term, and author_months all derive from the
    same commit-activity log, a reviewer should ask whether the low MMRE reflects genuine
    predictive signal or shared measurement source. We checked all three key correlations:"""),
    data_table(
        ["Feature", "Spearman r with author_months", "Note"],
        [
            ["contributor_count",      "+0.705", "High — both derived from who committed"],
            ["contributors_x_freq",    "+0.650", "High — interaction amplifies the overlap"],
            ["commit_frequency",       "+0.395", "Moderate"],
        ],
        col_widths=[5*cm, 5*cm, 6.6*cm]
    ),
    p("""This is <b>construct overlap, not train/test leakage.</b> Strict leakage means
    test-set information bleeding into training, inflating CV scores — the CV here is clean.
    What exists is that author_months deduplicates to unique person-days and divides by 20,
    so it is strongly related to — but not a deterministic function of — contributor count.
    A 50-person team each committing once total has very different author_months than 50 people
    each active for months. The r = 0.70 says they are strongly correlated, not identical."""),
    sp(),
    p("""The low MMRE therefore reflects a predictive association that is entangled with a
    shared measurement source; it does not establish that team size causes effort. Jira-tracked
    hours or another independent ground truth would permit a cleaner accuracy evaluation."""),
    callout("""An earlier version of this section reported Spearman(commit_frequency, author_months) = 0.04
    and concluded "no meaningful leakage exists." That figure was computed with Pearson correlation,
    not Spearman. The corrected Spearman values are shown in the table above. The conclusion changes:
    construct overlap exists in the dominant feature (r = 0.70), and the MMRE should be interpreted
    accordingly — as a strong proof-of-concept, not as a clean accuracy benchmark.""", AMBER_BG, AMBER),
    sp(),

    p("7.4 Popularity Hold-out (Not Temporal Validation)", H2),
    p("""The current CSV has no trustworthy project-date field, so a chronological test cannot
    be claimed. The available sensitivity check sorts repositories by star count. Stars measure
    popularity, not time:"""),
    bullet("Train on the higher-star partition"),
    bullet("Test on the lower-star partition"),
    bullet("Select hyperparameters with inner five-fold CV on the training partition only"),
    bullet("Re-fit scalers and supervised correlation weights entirely on training data"),
    sp(),
    data_table(
        ["Validation method", "MMRE", "PRED(25)", "Notes"],
        [
            ["Nested 10x5 CV", f"{G_CV['mmre']:.4f}", f"{G_CV['pred25']:.1%}", "Headline GitHub estimate"],
            ["Popularity hold-out", f"{POPULARITY['mmre']:.4f}", f"{POPULARITY['pred25']:.1%}", f"High-star train to low-star test (n={POPULARITY['test_n']})"],
            ["COCOMO-81 nested CV", f"{C_CV['mmre']:.4f}", f"{C_CV['pred25']:.1%}", "Separate historical benchmark"],
        ],
        col_widths=[4.5*cm, 2.5*cm, 2.5*cm, 7.1*cm]
    ),
    callout("""This result is a popularity sensitivity check only. It must not be described as
    temporal generalisation. Future collection records repository creation and sampled commit
    dates so a genuinely chronological evaluation can be performed.""", AMBER_BG, AMBER),
]


# ── SECTION 8: VALIDATION ─────────────────────────────────────────────────────
story += [
    p("8. Validation Results", H1), rule(BLUE_BDR),

    p("8.1 Ablation Study — Does Training-Derived Weighting Help?", H2),
    p("""A fair ablation must give the unweighted baseline its OWN hyperparameter search.
    Reusing the C/gamma/epsilon tuned for the weighted feature space disadvantages the
    baseline and inflates the apparent benefit. Below, each variant is independently
    tuned over the same grid, with leakage-free nested evaluation:"""),
    data_table(
        ["Model variant", "Nested-CV MMRE", "Weighting comparison"],
        [
            ["Plain SVR (unweighted, independently tuned)", f"{C_FAHP['unweighted_mmre']:.4f}", "baseline"],
            ["Proxy-weighted SVR", f"{C_FAHP['weighted_mmre']:.4f}", f"{C_FAHP['effect_pct']:.1f}% lower MMRE in this comparison"],
        ],
        col_widths=[8*cm, 3*cm, 5.6*cm]
    ),
    p(f"""In matched nested validation, training-derived proxy weighting changes COCOMO-81 MMRE by
    <b>{C_FAHP['effect_pct']:.1f}%</b> ({C_FAHP['unweighted_mmre']:.4f} to
    {C_FAHP['weighted_mmre']:.4f}). This is a sample-specific comparison, not proof that
    weighting improves every future dataset."""),
    p(f"""Across the {EX_REPEAT_ACCURACY['repeat_count']} repeated 5-outer/3-inner runs used
    for the explainability study, the weighted model has lower MMRE in
    {EX_REPEAT_ACCURACY['weighted_better_repeat_count']}/{EX_REPEAT_ACCURACY['repeat_count']}
    repeats. Median weighted MMRE is {EX_REPEAT_ACCURACY['weighted_mmre']['median']:.6f}
    [{EX_REPEAT_ACCURACY['weighted_mmre']['minimum']:.6f}, {EX_REPEAT_ACCURACY['weighted_mmre']['maximum']:.6f}],
    versus {EX_REPEAT_ACCURACY['unweighted_mmre']['median']:.6f}
    [{EX_REPEAT_ACCURACY['unweighted_mmre']['minimum']:.6f}, {EX_REPEAT_ACCURACY['unweighted_mmre']['maximum']:.6f}]
    unweighted. The median paired difference (unweighted minus weighted) is
    {EX_REPEAT_ACCURACY['paired_mmre_difference_unweighted_minus_weighted']['median']:+.6f}
    [{EX_REPEAT_ACCURACY['paired_mmre_difference_unweighted_minus_weighted']['minimum']:+.6f},
    {EX_REPEAT_ACCURACY['paired_mmre_difference_unweighted_minus_weighted']['maximum']:+.6f}].
    Median MdMRE is {EX_REPEAT_ACCURACY['weighted_mdmre']['median']:.6f} weighted versus
    {EX_REPEAT_ACCURACY['unweighted_mdmre']['median']:.6f} unweighted, with paired difference
    {EX_REPEAT_ACCURACY['paired_mdmre_difference_unweighted_minus_weighted']['median']:+.6f}.
    Median MAE is {EX_REPEAT_ACCURACY['weighted_mae_person_months']['median']:.6f} versus
    {EX_REPEAT_ACCURACY['unweighted_mae_person_months']['median']:.6f} person-months, with paired
    difference {EX_REPEAT_ACCURACY['paired_mae_difference_unweighted_minus_weighted_person_months']['median']:+.6f} PM.
    Repeated partitions overlap and are not independent inferential units, so this is descriptive;
    no uncorrected significance test is claimed."""),
    sp(),
    p(f"""<b>GitHub comparison:</b> the independently tuned unweighted model scores
    {G_FAHP['unweighted_mmre']:.4f}, while the weighted model scores
    {G_FAHP['weighted_mmre']:.4f}, a {G_FAHP['effect_pct']:.1f}% lower MMRE in these
    nested folds. The result is reported as a comparison because it does not establish
    a universal causal improvement from feature weighting."""),
    sp(),

    p("8.2 Cross-Dataset Generalisation (NASA-93)", H2),
    p("""To test whether the COCOMO-81 model generalises beyond its training distribution,
    we evaluate it on NASA-93 — 93 entirely different NASA software projects collected
    12 years later. No NASA-93 data was used in training or tuning."""),
    data_table(
        ["Evaluation", "Dataset", "n", "MMRE", "PRED(25)"],
        [
            ["Nested 10x5 CV", "COCOMO-81", str(C['n']), f"{C_CV['mmre']:.4f}", f"{C_CV['pred25']:.1%}"],
            ["External test", "NASA-93", str(NASA['n']), f"{NASA['mmre']:.4f}", f"{NASA['pred25']:.1%}"],
        ],
        col_widths=[4*cm, 3.5*cm, 1*cm, 2.5*cm, 3.6*cm]
    ),
    p(f"""MMRE is {C_CV['mmre']:.4f} in nested COCOMO-81 validation and
    {NASA['mmre']:.4f} on NASA-93. The external result measures transfer to a separate
    dataset; it does not guarantee performance for modern commercial projects."""),
    sp(),

    p("8.3 The 80% Prediction Interval", H2),
    p("""Beyond point estimates, the tool shows an 80% prediction interval — a range
    where approximately 4 out of 5 real projects should fall. This is computed
    empirically from pure nested outer-fold residuals (no parametric assumption):"""),
    formula_box("error = (prediction - actual) / actual<br/>PI = [prediction / (1 + q90), prediction / (1 + q10)]"),
    p(f"""For COCOMO-81, q10={C_PI['q10']:+.3f} and q90={C_PI['q90']:+.3f}.
    If the point prediction is 100 PM, the transformed 80% interval is
    [{100/(1+C_PI['q90']):.1f}, {100/(1+C_PI['q10']):.1f}] PM. Roughly one in five
    comparable projects may fall outside this empirical range."""),
    sp(),

    p("8.4 Per Development Mode Breakdown", H2),
    p("""Not all projects are equally predictable. Splitting the COCOMO-81 results
    by development mode reveals an interesting pattern:"""),
    data_table(
        ["Mode", "n", "Nested-fold MMRE", "PRED(25)", "Notes"],
        [
            ["Organic", str(C['mode_metrics']['organic']['n']), f"{C['mode_metrics']['organic']['mmre']:.4f}", f"{C['mode_metrics']['organic']['pred25']:.1%}", "Small familiar projects"],
            ["Semi-detached", str(C['mode_metrics']['semidetached']['n']), f"{C['mode_metrics']['semidetached']['mmre']:.4f}", f"{C['mode_metrics']['semidetached']['pred25']:.1%}", "Mixed experience"],
            ["Embedded", str(C['mode_metrics']['embedded']['n']), f"{C['mode_metrics']['embedded']['mmre']:.4f}", f"{C['mode_metrics']['embedded']['pred25']:.1%}", "Hard real-time or constrained"],
        ],
        col_widths=[3.5*cm, 1*cm, 2.5*cm, 2.5*cm, 7.1*cm]
    ),
    p("""These subsets are small, especially semi-detached (n=11). Their metrics are
    descriptive breakdowns of the outer-fold predictions, not independent validation studies."""),
    callout("""<b>Takeaway:</b> Use per-mode numbers as context, not a strict trust ranking.
    Sampling variation can be substantial at these group sizes.""", AMBER_BG, AMBER),
]


# ── SECTION 9: WORKED EXAMPLE ─────────────────────────────────────────────────
story += [
    p("9. End-to-End Worked Example", H1), rule(BLUE_BDR),
    p("""Let's walk through a complete prediction for a realistic project:
    a <b>FinTech payment processing API</b>, 40 KLOC, Semi-detached mode,
    with a moderately experienced team, some schedule pressure, and Copilot adopted."""),
    sp(),

    p("Step 1 — Gather Inputs", H2),
    data_table(
        ["Input", "Value", "Rationale"],
        [
            ["LOC",  "40 KLOC", "Mid-size API service"],
            ["RELY", "High (1.15)", "Financial transactions — SLA breach = money loss"],
            ["DATA", "Nominal (1.00)", "Standard database complexity"],
            ["CPLX", "High (1.15)", "Real-time settlement logic"],
            ["TIME", "Nominal (1.00)", "Standard latency requirements"],
            ["STOR", "Nominal (1.00)", "Cloud deployment, no memory pressure"],
            ["VIRT", "Low (0.87)", "Stable cloud infrastructure"],
            ["TURN", "Low (0.87)", "Fast CI/CD pipeline"],
            ["ACAP", "High (0.86)", "Strong architects"],
            ["AEXP", "Nominal (1.00)", "Some FinTech experience"],
            ["PCAP", "High (0.86)", "Solid developer team"],
            ["VEXP", "High (0.90)", "AWS-experienced team"],
            ["LEXP", "Nominal (1.00)", "TypeScript / Node.js — familiar"],
            ["MODP", "High (0.91)", "Good Agile practices"],
            ["TOOL", "High (0.91)", "Copilot + good tooling"],
            ["SCED", "High (1.04)", "Fixed product launch date"],
            ["Dev Mode", "Semi-detached", "Mixed team, moderate constraints"],
        ],
        col_widths=[2.5*cm, 3*cm, 11.1*cm]
    ),
    sp(),

    p("Step 2 — Compute the Raw COCOMO Estimate", H2),
    formula_box("EAF = 1.15 × 1.00 × 1.15 × 1.00 × 1.00 × 0.87 × 0.87 × 0.86 × 1.00 × 0.86 × 0.90 × 1.00 × 0.91 × 0.91 × 1.04"),
    formula_box("EAF ≈ 0.574   (favourable team/tools conditions reducing the baseline)"),
    formula_box("Baseline (Semi-det) = 3.0 × 40^1.12 = 3.0 × 62.3 = 186.9 PM\nCOCOMO Effort = 186.9 × 0.574 ≈ 107 PM"),
    sp(),

    p("Step 3 — SVR Prediction (What the Model Actually Returns)", H2),
    p("""The SVR doesn't use the COCOMO formula directly. It:"""),
    bullet("Converts the raw inputs (LOC, 15 multipliers, dev mode ordinal) to a feature vector"),
    bullet("Log-transforms, MinMax-scales, and applies the training-derived proxy weights"),
    bullet("Passes it through the trained RBF kernel SVR"),
    bullet("Inverse-scales and expm1 transforms back to person-months"),
    p("""For these exact inputs the model returns <b>111 PM</b> — nearly identical to the
    COCOMO formula's 107 PM. This is expected: with a well-calibrated EAF (0.574) on a
    Semi-detached project squarely within the training distribution, the SVR and the
    COCOMO multiplicative formula converge. The SVR adds value on projects where
    non-linear interactions between drivers matter more — here, the drivers are
    moderate and the two methods agree."""),
    sp(),

    p("Step 4 — Illustrative Local SHAP Snapshot (seed=42, fixed background n=50)", H2),
    p("""Background: the 50 training projects with effort ≤ 600 PM, selected for a responsive
    display explanation. Because that selection uses the outcome, this background is illustrative
    rather than representative. The resulting base value in the frozen run is <b>115.8 PM</b>,
    and the SHAP values are model-output attributions in PM units. This frozen example had
    0.00 PM stored additivity error; it is not the repeated OOF experiment in Section 5."""),
    sp(),
    data_table(
        ["Feature", "SHAP value", "Reading"],
        [
            ["LOC (40 KLOC)",  "+69.7 PM",  "Raises this model output relative to the fixed background"],
            ["RELY (High)",    "+14.0 PM",  "Raises this model output relative to the fixed background"],
            ["DevMode (Semi)", "+7.6 PM",   "Raises this model output relative to the fixed background"],
            ["TOOL (High)",    "−23.5 PM",  "Lowers this model output relative to the fixed background"],
            ["TIME (Nominal)", "−21.8 PM",  "Lowers this model output relative to the fixed background"],
            ["TURN (Low)",     "−15.1 PM",  "Lowers this model output relative to the fixed background"],
            ["PCAP (High)",    "−12.4 PM",  "Lowers this model output relative to the fixed background"],
            ["VEXP (High)",    "−11.3 PM",  "Lowers this model output relative to the fixed background"],
            ["ACAP (High)",    "−7.7 PM",   "Lowers this model output relative to the fixed background"],
            ["MODP (High)",    "−4.6 PM",   "Lowers this model output relative to the fixed background"],
            ["All others",     "0.0 PM",    "Negligible attribution for this input/background/model combination"],
            ["Baseline",       "115.8 PM",  "Expected model output over the selected 50-project display background (seed=42)"],
            ["<b>Total</b>",   "<b>110.8 PM</b>", "Frozen illustrative run: 115.8 − 5.0 = 110.8; rounded additivity residual 0.00%"],
        ],
        col_widths=[3.5*cm, 2.5*cm, 10.6*cm]
    ),
    sp(),

    p("Step 5 — Indicative Schedule and Staffing Ratio", H2),
    formula_box("Calendar months = 2.5 × 111^0.38 = 2.5 × 5.98 ≈ 15 months"),
    formula_box("Implied average staffing = 111 / 15 ≈ 7 engineers"),
    sp(),

    p("Step 6 — Post-hoc GenAI Sensitivity Scenario", H2),
    formula_box("Scenario effort = 111 × 0.78 = 87 PM   (assumed 22% multiplier)"),
    formula_box("AI calendar = 2.5 × 87^0.38 = 2.5 × 5.46 ≈ 14 months   |   AI team = 6 engineers"),
    formula_box(f"80% PI: 111 / (1 + {C_PI['q90']:+.3f}) = {111/(1+C_PI['q90']):.0f} PM  to  111 / (1 + {C_PI['q10']:+.3f}) = {111/(1+C_PI['q10']):.0f} PM"),
    callout(f"""<b>Summary for the FinTech API project (illustrative model output):</b><br/>
    SVR prediction: 110.8 PM  ·  COCOMO: 107 PM  ·  estimates are close for this illustrative input<br/>
    Without AI: 111 PM, 15 months, 7 engineers<br/>
    With the assumed Copilot adjustment: 87 PM, 14 months, 6 engineers (modelled difference ~24 PM = 22%; not a causal forecast)<br/>
    80% empirical interval: {111/(1+C_PI['q90']):.0f}–{111/(1+C_PI['q10']):.0f} PM (wide — reflects genuine estimation uncertainty)<br/>
    SHAP baseline: 115.8 PM (expected model output over the selected 50-project display background, seed=42)<br/>
    Largest positive local attribution: LOC (+69.7 PM); largest negative local attributions:
    TOOL (−23.5 PM), TIME (−21.8 PM)""",
    GREEN_BG, GREEN),
]


# ── SECTION 10: LIMITATIONS ───────────────────────────────────────────────────
story += [
    p("10. Limitations and Scope of Validity", H1), rule(BLUE_BDR),
    p("""No model is perfect. Being transparent about limitations is a sign of
    rigorous research, not weakness. Use these estimates as structured guidance,
    not budget commitments."""),
    sp(),

    p("10.1 COCOMO-81 Model", H2),
    bullet("<b>Training data age:</b> 63 projects from the 1980s, mostly defense/aerospace. The model has never seen React, microservices, CI/CD, or cloud-native development."),
    bullet("<b>LOC as a proxy:</b> Modern teams often work with story points, not LOC. The model accepts LOC as a proxy for scope — use sprint velocity × total sprints as an approximation."),
    bullet("<b>Not a reproduction:</b> Sehra et al. (2019) used FAHP-weighted LSSVM; this implementation uses sklearn epsilon-SVR with a mathematically analogous weighted RBF transformation."),
    bullet("<b>Weight derivation:</b> Independent expert pairwise comparisons are replaced with a supervised, data-derived FAHP-inspired proxy. A faithful replication-and-extension study requires the original judgements or a documented new expert panel."),
    bullet(f"<b>Proxy sparsity and consistency:</b> Full-data Chang extent weights are nonzero for {EX_PROXY_DIAGNOSTICS['effort_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} effort projects, {EX_PROXY_DIAGNOSTICS['loc_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} LOC projects, and {EX_PROXY_DIAGNOSTICS['combined_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} combined weights. The effort/LOC pairwise consistency ratios are {EX_PROXY_DIAGNOSTICS['effort_pairwise_consistency_ratio']:.6f}/{EX_PROXY_DIAGNOSTICS['loc_pairwise_consistency_ratio']:.6f}; proxy sensitivity and genuine expert elicitation remain necessary for confirmation."),
    bullet("<b>SHAP scope:</b> Attributions are relative to a chosen training background and are not causal evidence. Correlated COCOMO drivers may redistribute attribution; dependence-aware sensitivity analysis remains future work."),
    bullet("<b>Size-response compression.</b> The model's effort response to project size is slightly sublinear — the effective in-range exponent is ~0.88, versus COCOMO-81's 1.05 — a consequence of RBF-kernel smoothing pulling predictions toward the training distribution. Within the training range this is minor, but the compression intensifies with distance from the data: beyond the largest training projects it produces systematic under-prediction (the basis for the extrapolation warning at 73 KLOC in Organic mode). Large-project estimates should therefore be read as lower bounds, and inputs beyond the per-mode training range treated as unreliable rather than merely uncertain."),
    sp(),

    p("10.2 GitHub Model", H2),
    bullet("<b>Effort proxy quality:</b> Author-months from commit logs ≠ actual billed hours. Open-source volunteers often commit in bursts; paid developers have more consistent patterns."),
    bullet(f"<b>Sampling cap:</b> At most {PROXY['commit_sample_limit']:,} commits are sampled per repository (approximately {PROXY['approximate_cap_author_months']:.0f} author-months). Predictions above that range are extrapolated and not validated."),
    bullet("<b>Open-source bias:</b> GitHub repos are maintained by volunteers with different motivations than commercial development teams. Coordination overhead and rework patterns may differ."),
    bullet("<b>Feature availability:</b> AI tool adoption is inferred from config files — not all teams configure their tools in discoverable ways, leading to underestimation of adoption."),
    sp(),

    p("10.3 GenAI Multipliers", H2),
    bullet("<b>Post-hoc adjustment:</b> Multipliers are applied to total effort. AI tools primarily help with coding-specific tasks (code completion, test generation); planning, stakeholder management, and architecture work may see less benefit."),
    bullet("<b>Study variation:</b> The illustrative range is wide and mixes evidence types. Treat it as sensitivity analysis, not a planning estimate; validate any effect on independent completed-project data."),
    bullet("<b>COCOMO TOOL interaction:</b> The TOOL driver already accounts for some tooling quality. Using both TOOL at Very High and a Heavy AI multiplier may double-count some benefit."),
    sp(),

    p("10.4 What the Model Cannot Do", H2),
    bullet("Predict individual sprint velocity or week-by-week burn rate"),
    bullet("Account for organisational dysfunction, communication failures, or political delays"),
    bullet("Estimate effort for research-phase work where requirements are undefined"),
    bullet("Replace expert estimation — it is a structured input to a human decision, not a replacement for one"),
]


# ── SECTION 11: MITACS VISION ─────────────────────────────────────────────────
story += [
    p("11. The Mitacs Vision — Why Company Data Matters", H1), rule(BLUE_BDR),
    p("""Every model in this tool is limited by the data it was trained on.
    COCOMO-81 learned from 1980s defense software. The GitHub model learned from
    open-source volunteers. <b>Neither knows your team.</b>"""),
    sp(),
    p("""A model trained on your company's completed projects — with actual Jira-tracked
    hours as the ground truth — would:"""),
    bullet("Learn your team's specific velocity, not a 1981 average"),
    bullet("Identify which COCOMO drivers actually predict overruns <i>for you</i>"),
    bullet("Discover your chronically underestimated project types"),
    bullet("Estimate AI-tool associations using independently recorded adoption and completed-project effort"),
    sp(),

    p("11.1 What Can Be Claimed Today", H2),
    data_table(
        ["Model", "Training data", "Validation MMRE", "PRED(25)"],
        [
            ["COCOMO-81 benchmark", f"{C['n']} projects", f"{C_CV['mmre']:.4f} nested CV", f"{C_CV['pred25']:.1%}"],
            ["GitHub proof of concept", f"{G['n']} repos", f"{G_CV['mmre']:.4f} nested CV", f"{G_CV['pred25']:.1%}"],
            ["Your company data", "Not yet provided", "Not measured", "Not measured"],
        ],
        col_widths=[5.5*cm, 4.5*cm, 3*cm, 3.6*cm]
    ),
    p("""Company data could be more representative of the intended use case, but no numerical
    improvement is promised. Its MMRE and PRED(25) must be measured using held-out completed
    projects after the target definition and data quality are validated."""),
    sp(),

    p("11.2 What You Would Provide", H2),
    bullet("<b>From Jira:</b> Story points, sprint velocity, cycle time, bug reopen rate, team size, planned vs actual delivery"),
    bullet("<b>From GitHub:</b> PRs merged, review time, test coverage, language count, AI tool adoption, commit frequency"),
    bullet("<b>Scale:</b> A small upload can be checked for schema and feasibility, but reliable model validation generally needs substantially more completed projects; uncertainty must be reported for the actual sample."),
    sp(),

    p("11.3 The 4-Month Deliverable", H2),
    data_table(
        ["Phase", "Timeline", "Deliverable"],
        [
            ["Data collection & feature engineering", "Month 1–2", "Clean dataset with effort ground truth + feature set"],
            ["Model training & validation",           "Month 3",   "Weighted-SVR adaptation, repeated OOF KernelSHAP, nested validation and chronological hold-out when dates permit"],
            ["Deployment & handoff",                  "Month 4",   "Live interactive dashboard on your internal infrastructure, documentation, maintenance guide — no sensitive data leaves your environment"],
        ],
        col_widths=[6*cm, 2.5*cm, 8.1*cm]
    ),
    sp(),

    p("11.4 Cost Structure", H2),
    data_table(
        ["Contributor", "Amount", "What it covers"],
        [
            ["Industry partner", "$7,500", "4 months of focused research and development"],
            ["Mitacs (federal funding)", "$7,500", "Matched 1:1 — equal federal and industry contribution"],
            ["Total research value", "$15,000", "Custom AI estimation tool delivered in 4 months"],
        ],
        col_widths=[4*cm, 3*cm, 9.6*cm]
    ),
]


# ── SECTION 12: REFERENCES ────────────────────────────────────────────────────
story += [
    p("12. References", H1), rule(BLUE_BDR),
    data_table(
        ["Reference", "Relevance"],
        [
            ["Boehm, B.W. (1981). Software Engineering Economics. Prentice Hall.", "COCOMO-81 model, cost drivers, EAF"],
            ["Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.", "Foundation of AHP"],
            ["Buckley, J.J. (1985). Fuzzy hierarchical analysis. Fuzzy Sets and Systems, 17(3), 233–247.", "Fuzzy TFN scale for pairwise comparisons"],
            ["Chang, D.Y. (1996). Applications of the extent analysis method on fuzzy AHP. European Journal of Operational Research, 95(3), 649–655.", "Extent-analysis component used in the data-derived proxy"],
            ["Sehra, S.K., Brar, Y.S., Kaur, N., &amp; Sehra, S.S. (2019). Software effort estimation using FAHP and weighted kernel LSSVM machine. Soft Computing, 23(21), 10881–10900. https://doi.org/10.1007/s00500-018-3639-2", "Foundational FAHP-weighted LSSVM study; this project is an adaptation, not a reproduction"],
            ["Peng, S. et al. (2023). The impact of AI coding assistants on developer productivity: A controlled experiment. GitHub Research.", "55% faster task completion in the controlled experiment"],
            ["McKinsey Global Institute (2023). The economic potential of generative AI.", "20–45% developer productivity gain"],
            ["Kalliamvakou, E. (2022). Research: Quantifying GitHub Copilot's impact on developer productivity. GitHub Blog.", "Copilot PR merge speed study"],
            ["Lundberg, S.M., Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.", "SHAP explainability framework"],
            ["Najm, A., Zakrani, A., &amp; Marzak, A. (2023). Efficient Shapely Explanation of Support Vector Regression for Agile and Non-agile Software Effort Estimation. Intelligent Sustainable Systems, 711–729. https://doi.org/10.1007/978-981-19-7663-6_67", "Prior model-agnostic explanation of SVR effort estimates"],
            ["Jayadi, P., Prasetya, D.D., &amp; Widiyaningtyas, T. (2025). Explanation of a Simple Machine Learning Model for Software Effort Estimation using SHAP (SHapley Additive exPlanations). Proc. 9th ICEEIE, 1–5. https://doi.org/10.1109/ICEEIE66203.2025.11254827", "Prior SHAP application in software-effort estimation"],
            ["Stack Overflow Developer Survey (2024). stackoverflow.com/survey", "AI tool adoption statistics"],
            ["GitHub Octoverse (2024). github.com/octoverse", "Copilot code contribution data"],
        ],
        col_widths=[9.5*cm, 7.1*cm]
    ),
]


# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=ML, rightMargin=MR,
    topMargin=MT,  bottomMargin=MB,
    title="Software Effort Estimator — Technical Documentation",
    author="Keshav",
    subject="Sehra et al. (2019)-inspired weighted RBF-SVR and repeated out-of-fold KernelSHAP analysis",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
shutil.copyfile(OUT, PROJECT_OUT)

html_explainability_rows = "".join(
    "<tr>"
    f"<td>{row['feature']}</td>"
    f"<td>{row['mean_weighting_proxy']:.6f} (#{row['proxy_rank']})</td>"
    f"<td>{row['unweighted_oof_shap_reliance']:.6f} (#{row['unweighted_shap_rank']})</td>"
    f"<td>{row['weighted_oof_shap_reliance']:.6f} (#{row['weighted_shap_rank']})</td>"
    "</tr>"
    for row in EX_TOP_FEATURES
)

html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Software Effort Estimator — Audited Technical Summary</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;color:#171717;max-width:980px;margin:3rem auto;padding:0 1.5rem}}
h1,h2{{line-height:1.2}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #ddd;padding:.55rem;text-align:left}} th{{background:#f5f5f5}}
.note{{border-left:4px solid #d97706;background:#fffbeb;padding:1rem}} code{{background:#f5f5f5;padding:.15rem .3rem}}
</style></head><body>
<h1>Software Effort Estimator</h1><p>Audited technical summary generated from <code>metrics.json</code>.</p>
<h2>Validation results</h2><table><thead><tr><th>Model</th><th>n</th><th>Method</th><th>MMRE</th><th>PRED(25)</th><th>PRED(50)</th></tr></thead><tbody>
<tr><td>COCOMO-81</td><td>{C['n']}</td><td>Nested 10 outer / 5 inner folds</td><td>{C_CV['mmre']:.4f}</td><td>{C_CV['pred25']:.1%}</td><td>{C_CV['pred50']:.1%}</td></tr>
<tr><td>GitHub proof of concept</td><td>{G['n']}</td><td>Nested 10 outer / 5 inner folds</td><td>{G_CV['mmre']:.4f}</td><td>{G_CV['pred25']:.1%}</td><td>{G_CV['pred50']:.1%}</td></tr>
<tr><td>NASA-93 external test</td><td>{NASA['n']}</td><td>Never used for selection</td><td>{NASA['mmre']:.4f}</td><td>{NASA['pred25']:.1%}</td><td>—</td></tr>
<tr><td>GitHub popularity hold-out</td><td>{POPULARITY['test_n']} test</td><td>High-star train → low-star test</td><td>{POPULARITY['mmre']:.4f}</td><td>{POPULARITY['pred25']:.1%}</td><td>—</td></tr>
</tbody></table>
<p class="note"><strong>Popularity is not time.</strong> The current CSV lacks reliable dates, so the hold-out must not be described as temporal validation. Model selection used only the training partition.</p>
<h2>Training-derived weighting comparison</h2><table><thead><tr><th>Dataset</th><th>Unweighted MMRE</th><th>Weighted MMRE</th><th>Difference</th></tr></thead><tbody>
<tr><td>COCOMO-81</td><td>{C_FAHP['unweighted_mmre']:.4f}</td><td>{C_FAHP['weighted_mmre']:.4f}</td><td>{C_FAHP['effect_pct']:.1f}% lower</td></tr>
<tr><td>GitHub</td><td>{G_FAHP['unweighted_mmre']:.4f}</td><td>{G_FAHP['weighted_mmre']:.4f}</td><td>{G_FAHP['effect_pct']:.1f}% lower</td></tr>
</tbody></table><p>These are matched nested-validation comparisons, not a claim of universal improvement.</p>
<h2>Repeated out-of-fold KernelSHAP pilot</h2>
<p>The leakage-safe explainability protocol uses {EX_PROTOCOL['repeats']} repeats of {EX_PROTOCOL['outer_splits']} outer folds with {EX_PROTOCOL['inner_splits']} inner folds. Only outer-test projects are explained. Each fold uses all outer-training projects as its background, <code>nsamples={EX_PROTOCOL['nsamples']}</code>, and <code>l1_reg={EX_PROTOCOL['l1_reg']}</code>; all 17 features enter the solution. Maximum stored additivity error is {EXPLAIN['max_additivity_error_person_months']:.6f} person-months.</p>
<table><thead><tr><th>Comparison</th><th>Spearman rho</th><th>Kendall tau-b</th><th>JS similarity</th><th>Top-3 overlap</th></tr></thead><tbody>
<tr><td>Primary: training-derived FAHP-inspired proxy vs unweighted OOF SHAP reliance</td><td>{EX_PRIMARY['spearman_rho']:.6f}</td><td>{EX_PRIMARY['kendall_tau_b']:.6f}</td><td>{EX_PRIMARY['jensen_shannon_similarity']:.6f}</td><td>{EX_PRIMARY['top_k_intersection']}/3; Jaccard {EX_PRIMARY['top_k_jaccard']:.6f}</td></tr>
<tr><td>Weighted model (descriptive; partly induced)</td><td>{EX_WEIGHTED['spearman_rho']:.6f}</td><td>{EX_WEIGHTED['kendall_tau_b']:.6f}</td><td>{EX_WEIGHTED['jensen_shannon_similarity']:.6f}</td><td>{EX_WEIGHTED['top_k_intersection']}/3; Jaccard {EX_WEIGHTED['top_k_jaccard']:.6f}</td></tr>
</tbody></table>
<p>Across repeats, the primary Spearman median is {EX_PRIMARY_REPEATS['spearman_rho']['median']:.6f} [{EX_PRIMARY_REPEATS['spearman_rho']['minimum']:.6f}, {EX_PRIMARY_REPEATS['spearman_rho']['maximum']:.6f}]. A {EX_BOOTSTRAP['resamples']:,}-resample project-level percentile procedure gives a conditional {EX_BOOTSTRAP['coverage']:.0%} interval [{EX_BOOTSTRAP['metrics']['spearman_rho']['lower']:.6f}, {EX_BOOTSTRAP['metrics']['spearman_rho']['upper']:.6f}]; it does not refit models, folds, tuning, or the explainer. Joint repeat ranking stability has median Spearman {EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['median']:.6f} [{EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['minimum']:.6f}, {EX_UNWEIGHTED_GLOBAL_STABILITY['spearman_rho']['maximum']:.6f}] across {EX_UNWEIGHTED_GLOBAL_STABILITY['pair_count']} repeat pairs. Joint local-attribution cosine similarity has median {EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['median']:.6f} [{EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['minimum']:.6f}, {EX_UNWEIGHTED_LOCAL_STABILITY['cosine_similarity']['maximum']:.6f}] across {EX_UNWEIGHTED_LOCAL_STABILITY['comparison_count']} within-project repeat comparisons; split, model, background, and seed vary together.</p>
<p>The fixed-model numerical audit at {EX_PROTOCOL['nsamples']} versus {EX_NUMERICAL['reference_nsamples']} samples has global Spearman {EX_NUMERICAL_MAIN['global_agreement_to_1024_reference']['spearman_rho']:.6f}, Jensen–Shannon similarity {EX_NUMERICAL_MAIN['global_agreement_to_1024_reference']['jensen_shannon_similarity']:.6f}, identical top three, and median local cosine {EX_NUMERICAL_MAIN['local_cosine_to_1024_reference']['median']:.6f}. In the {EX_PERMUTATION['permutations']:,}-permutation feature-label benchmark, Spearman empirical p={EX_PERMUTATION['empirical_p_values']['spearman_rho']['empirical_p']:.6f} while top-3 Jaccard p={EX_PERMUTATION['empirical_p_values']['top_k_jaccard']['empirical_p']:.6f}; the narrow benchmark is descriptive, not a population hypothesis test.</p>
<p>Repeated accuracy is descriptive: weighting has lower MMRE in {EX_REPEAT_ACCURACY['weighted_better_repeat_count']}/{EX_REPEAT_ACCURACY['repeat_count']} runs. Median weighted versus unweighted MMRE is {EX_REPEAT_ACCURACY['weighted_mmre']['median']:.6f} versus {EX_REPEAT_ACCURACY['unweighted_mmre']['median']:.6f}; MdMRE is {EX_REPEAT_ACCURACY['weighted_mdmre']['median']:.6f} versus {EX_REPEAT_ACCURACY['unweighted_mdmre']['median']:.6f}; and MAE is {EX_REPEAT_ACCURACY['weighted_mae_person_months']['median']:.6f} versus {EX_REPEAT_ACCURACY['unweighted_mae_person_months']['median']:.6f} person-months. Partitions overlap, so the repeats are not independent inferential units and no uncorrected significance test is reported.</p>
<table><thead><tr><th>Feature</th><th>Proxy weight (rank)</th><th>Unweighted OOF SHAP (rank)</th><th>Weighted OOF SHAP (rank)</th></tr></thead><tbody>{html_explainability_rows}</tbody></table>
<p class="note"><strong>Interpretation limit:</strong> SHAP attributes model output relative to a chosen background; it is not causal evidence. The proxy is data-derived rather than an independent expert prior, and weighted-model agreement is partly induced by construction. Full-data Chang extent weights are nonzero for {EX_PROXY_DIAGNOSTICS['effort_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} effort, {EX_PROXY_DIAGNOSTICS['loc_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} LOC, and {EX_PROXY_DIAGNOSTICS['combined_project_weights_nonzero']}/{EX_PROXY_DIAGNOSTICS['project_count']} combined project weights; expert elicitation remains required for a confirmatory FAHP extension. Prior work already applied SHAP to SVR and simpler software-effort models (DOIs <a href="https://doi.org/10.1007/978-981-19-7663-6_67">10.1007/978-981-19-7663-6_67</a> and <a href="https://doi.org/10.1109/ICEEIE66203.2025.11254827">10.1109/ICEEIE66203.2025.11254827</a>).</p>
<p>Foundational study: Sehra, S. K., Brar, Y. S., Kaur, N., &amp; Sehra, S. S. (2019). <em>Software effort estimation using FAHP and weighted kernel LSSVM machine</em>. <em>Soft Computing, 23</em>(21), 10881–10900. <a href="https://doi.org/10.1007/s00500-018-3639-2">https://doi.org/10.1007/s00500-018-3639-2</a>. This project is an adaptation, not a reproduction.</p>
<h2>Prediction interval</h2><p>Signed relative error is <code>(prediction - actual) / actual</code>. The empirical interval is <code>[prediction / (1 + q90), prediction / (1 + q10)]</code>, using pure outer-fold residuals.</p>
<h2>GitHub effort proxy limitation</h2><p class="note">At most {PROXY['commit_sample_limit']:,} commits are sampled per repository, approximately {PROXY['approximate_cap_author_months']:.0f} author-months. Predictions above that cap are extrapolations and have not been validated.</p>
<h2>Your Data</h2><p>No accuracy is promised before data is supplied. The app validates CSV/XLSX schemas and provides a template; training is intentionally disabled until completed-project outcomes and sufficient sample size are available.</p>
</body></html>"""
with open(HTML_OUT, "w", encoding="utf-8") as html_file:
    html_file.write(html_report)

print(f"PDF saved: {OUT}")
print(f"PDF saved: {PROJECT_OUT}")
print(f"HTML saved: {HTML_OUT}")
