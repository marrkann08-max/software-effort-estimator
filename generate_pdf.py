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
    canvas.drawString(ML, MB - 14, "FAHP · Weighted-Kernel SVR · SHAP · Mitacs Accelerate Research")
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
     Paragraph("FAHP (Chang 1996) + Weighted-Kernel SVR (Sehra et al. 2018) + SHAP", BODY)],
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
    p("""This tool builds on that foundation and extends it in three ways:"""),
    bullet("<b>Better feature weighting</b> — Fuzzy AHP (FAHP) assigns each cost driver a mathematically derived importance score, so the model knows which factors matter most for your type of project."),
    bullet("<b>Non-linear learning</b> — Support Vector Regression (SVR) with a weighted kernel learns patterns from historical data and is evaluated separately from the original multiplicative COCOMO formula."),
    bullet("<b>Explainability</b> — SHAP values break down every prediction into per-feature contributions in person-months, so you know <i>why</i> the estimate is what it is."),
    sp(),
    callout("Think of it like this: COCOMO is a recipe from 1981 written for defence projects. This tool is the same recipe, upgraded with a modern oven, better ingredients, and a chef who explains every step.", BLUE_LIGHT, BLUE),
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
        col_widths=[2.8*cm, 6.5*cm, 1*cm, 1*cm, 5.3*cm]
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
        col_widths=[2.2*cm, 1.5*cm, 9*cm, 3.9*cm]
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
    p("""COCOMO treats all 15 cost drivers as equally important to the prediction task —
    it just multiplies them all together. But in reality, some features matter much more
    than others. <b>FAHP is how we discover and encode that importance.</b>"""),
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

    p("3.3 Chang's Extent Analysis (1996)", H2),
    p("""This model uses Chang's extent analysis to derive feature weights from the
    pairwise comparison matrices. The algorithm has three steps:"""),
    sp(),
    bullet("<b>Step 1 — Build pairwise matrices:</b> For each pair of features (i, j), compute the importance ratio r = corr(feature_i, effort) / corr(feature_j, effort) using Spearman correlations with the training data. Convert r to a TFN. Build an n×n matrix of TFNs.", "1."),
    sp(4),
    bullet("<b>Step 2 — Synthetic extent values:</b> For each feature i, sum across all j to get a fuzzy row sum S_i = (l_i, m_i, u_i). Normalise to get the synthetic extent value.", "2."),
    sp(4),
    formula_box("S_i = (Σ_j l_ij / Σ_i Σ_j u_ij ,  Σ_j m_ij / Σ_i Σ_j m_ij ,  Σ_j u_ij / Σ_i Σ_j l_ij)"),
    sp(4),
    bullet("<b>Step 3 — Degree of possibility:</b> For each feature i, compute V(S_i ≥ S_j) for all j ≠ i. The weight of feature i is the minimum of these values, then normalise.", "3."),
    sp(4),
    formula_box("V(S_i ≥ S_j) = 1 if m_i ≥ m_j,   0 if l_j ≥ u_i,   else (l_j - u_i) / ((m_i - u_i) - (m_j - l_j))"),
    sp(4),
    formula_box("weight_i = min_{j≠i} V(S_i ≥ S_j)   →   normalise so Σ weights = 1"),
    sp(),

    p("3.4 Worked Example: 3 Features", H2),
    p("""Suppose we have 3 features with Spearman correlations with effort:
    CPLX (r=0.65), ACAP (r=0.50), LEXP (r=0.20). We build a 3×3 pairwise matrix:"""),
    sp(),
    data_table(
        ["", "CPLX", "ACAP", "LEXP"],
        [
            ["CPLX", "(1,1,1)",   "(1,2,3)",   "(3,5,7)"],
            ["ACAP", "(1/3,1/2,1)","(1,1,1)", "(1,3,5)"],
            ["LEXP", "(1/7,1/5,1/3)","(1/5,1/3,1)","(1,1,1)"],
        ],
        col_widths=[2*cm, 5*cm, 5*cm, 4.6*cm]
    ),
    p("""After running Chang's extent analysis, we might get weights like:"""),
    data_table(
        ["Feature", "FAHP weight", "Interpretation"],
        [
            ["CPLX", "0.52", "Most important — dominates prediction"],
            ["ACAP", "0.35", "Moderately important"],
            ["LEXP", "0.13", "Less important"],
        ],
        col_widths=[3*cm, 4*cm, 9.6*cm]
    ),
    callout("""<b>Key insight:</b> FAHP doesn't just rank features — it produces weights that are
    embedded directly into the SVR kernel. A feature with weight 0.52 will have its
    distances measured ~4× more carefully than one with weight 0.13. Whether this lowers
    prediction error is measured by the matched weighted/unweighted nested-validation comparison.""", BLUE_LIGHT, BLUE),
]


# ── SECTION 4: SVR ───────────────────────────────────────────────────────────
story += [
    p("4. Weighted-Kernel Support Vector Regression (SVR)", H1), rule(BLUE_BDR),
    p("""Once FAHP gives us feature weights, we need a model that can learn non-linear
    patterns from the 63 COCOMO-81 training projects. That model is SVR."""),
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

    p("4.3 The FAHP Weighting — The Core Innovation", H2),
    p("""Here is where FAHP and SVR connect. Instead of measuring distance equally
    across all features, we <b>stretch the feature space</b> by the FAHP weights.
    Features that matter more are measured more carefully:"""),
    formula_box("K_FAHP(x_i, x_j) = exp(-γ × ||√θ · (x_i - x_j)||²)"),
    p("""Where θ is the vector of FAHP weights and √θ is applied element-wise.
    This is equivalent to:"""),
    formula_box("x_weighted = x × √θ"),
    p("""<b>Concretely:</b> if ACAP has FAHP weight 0.35 and LEXP has weight 0.05,
    then the weight ratio is 0.35/0.05 = 7.0×. Because the kernel uses <i>squared</i>
    distance, the effective distance scaling factor is √7.0 = 2.6× — meaning a
    difference in ACAP is measured 2.6× more sensitively in distance terms than
    the same numerical difference in LEXP. The kernel "pays more attention"
    to the features that matter."""),
    sp(),

    p("4.4 The Full Training Pipeline", H2),
    data_table(
        ["Step", "What happens", "Why"],
        [
            ["1. Log transform", "x → log(x),  y → log(1 + y)", "Effort is right-skewed; log space makes it more symmetric"],
            ["2. Scale", "MinMaxScaler to [0, 1]", "SVR is sensitive to feature scale differences"],
            ["3. FAHP weight", "x_weighted = x_scaled × √(FAHP_weights)", "Embeds feature importance into kernel distance"],
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
    gamma and epsilon. Scaling and FAHP weights are also fitted only on the relevant training
    rows. Deployment parameters are selected separately after evaluation.""", AMBER_BG, AMBER),
]


# ── SECTION 5: SHAP ──────────────────────────────────────────────────────────
story += [
    p("5. SHAP Explainability", H1), rule(BLUE_BDR),
    p("""SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) answers:
    <i>"The model predicted 180 person-months — how much did each feature contribute?"</i>
    This matters because a number without a reason is useless for planning."""),
    sp(),

    p("5.1 The Shapley Value Concept", H2),
    p("""Shapley values come from cooperative game theory. Imagine each feature is a
    "player" contributing to the prediction "game". The Shapley value of a feature
    is its <b>average marginal contribution</b> across all possible orderings of features."""),
    sp(),
    p("""In plain English: SHAP asks "if we added feature X to the model after considering
    all possible subsets of other features, what would it add to the prediction on average?"
    This gives a fair, mathematically rigorous attribution."""),
    sp(),
    formula_box("prediction = base_value + Σ SHAP_value(feature_i)"),
    p("""The base value is the model's average prediction across all training projects.
    Each SHAP value is in the same units as the prediction (person-months), so you can say:
    "High CPLX added +23 PM above the baseline."  """),
    sp(),

    p("5.2 KernelExplainer for SVR", H2),
    p("""SVR doesn't have a closed-form SHAP solution like tree models do.
    This tool uses <b>KernelExplainer</b> (model-agnostic SHAP), which:"""),
    bullet("Samples 30 background projects from the training set"),
    bullet("Evaluates the full pipeline (log → scale → weight → SVR → inverse-scale → expm1) on 150 random feature coalitions"),
    bullet("Fits a weighted linear model to approximate SHAP values for the current project"),
    sp(),
    p("""The SHAP values are computed in the <b>original raw feature space</b>, so each
    value directly represents person-month impact — not some abstract scaled quantity."""),
    sp(),

    p("5.3 Reading SHAP Outputs", H2),
    data_table(
        ["Chart", "What to read"],
        [
            ["Feature Contributions bar", "Red bars = features pushing effort UP above baseline.\nBlue bars = features pushing effort DOWN. Bar length = magnitude in PM."],
            ["Waterfall chart", "Shows the cumulative path from baseline (avg PM) to your prediction. Each step adds or subtracts PM."],
            ["FAHP vs SHAP bar chart", "Compares the prior weights (what FAHP thought mattered) vs posterior importance (what the trained model actually learned). Divergence is a research finding."],
        ],
        col_widths=[4.5*cm, 12.1*cm]
    ),
    sp(),
    callout("""<b>Note on the baseline value:</b> The SHAP base value is the mean model prediction
    over the background dataset. This tool uses the 50 training projects with effort ≤ 600 PM
    (80th percentile) as the representative background (seed=42, stable across runs).
    The 13 excluded projects are 1980s defence mega-programs (up to 11,400 PM) whose inclusion
    would push the base to 620 PM — making every modern-project waterfall open at an unintuitive
    starting point 4–6× the final prediction.
    The representative background gives a base of <b>~116 PM</b>, close to the geometric mean
    of the training distribution (127 PM). SHAP values remain in person-months throughout;
    additivity verified at 0.00% error. See Section 9 for a worked example with real output.""",
    BLUE_LIGHT, BLUE),
    sp(),
    callout("""<b>Important caveat — Nominal ratings can still show large SHAP values.</b>
    SHAP is always relative to the <i>training mean</i>, not to a "zero effort" baseline.
    If the 63 COCOMO-81 training projects averaged High TIME constraints, then a project
    with Nominal TIME will show a <i>negative</i> SHAP value — not because Nominal TIME
    reduces effort in absolute terms, but because it is below the training average.
    Similarly, Dev Mode encoded as an ordinal can absorb large SHAP values if it is the
    strongest separator in the training data. Always read SHAP values as
    "relative to what the model typically sees", not as absolute effort drivers.""",
    AMBER_BG, AMBER),
]


# ── SECTION 6: GENAI ──────────────────────────────────────────────────────────
story += [
    p("6. GenAI Productivity Adjustment", H1), rule(BLUE_BDR),
    p("""COCOMO-81 was published in 1981. Its TOOL cost driver captures the quality
    of development tools — and at Very High, it reduces effort by at most 17%.
    That was the best tools could do in 1981. GitHub Copilot was released in 2022."""),
    sp(),

    p("6.1 The Research Evidence", H2),
    data_table(
        ["Study", "n", "Finding", "Effort impact"],
        [
            ["Peng et al. (2023) — GitHub", "95 developers", "Copilot: 55% faster on isolated coding task (controlled experiment) — full-project gain not measured by this study", "Task-level: −55%"],
            ["McKinsey Global Institute (2023)", "850 projects", "20–45% productivity gain depending on AI adoption depth (industry report)", "−20–45%"],
            ["Kalliamvakou (2022) — GitHub blog", "Copilot cohort", "Copilot users merged PRs 26% faster, reported higher satisfaction", "~−26% on PRs"],
            ["COCOMO-81 TOOL driver", "63 projects", "Very High tools: maximum 17% effort reduction", "max −17%"],
        ],
        col_widths=[4.5*cm, 2.5*cm, 7*cm, 2.6*cm]
    ),
    sp(),
    callout("""<b>The research gap:</b> No existing effort estimation model — including Sehra et al. (2018),
    which this tool implements — accounts for modern AI coding tools. COCOMO caps at 17%.
    Studies show 22–55%. Quantifying this gap empirically on your company's data is the
    novel Mitacs research contribution.""", PURPLE_BG, PURPLE),
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
    p("""The range is shown as a confidence band in the UI, reflecting the study variation.
    A "Heavy AI" team of 50 that would have needed 200 PM is estimated at
    200 × 0.62 = <b>124 PM</b>, saving 76 PM (38%). The 80% prediction interval
    propagates through: 200 × 0.55 = 110 PM to 200 × 0.70 = 140 PM."""),
    sp(),
    p("""<b>Important caveat:</b> These multipliers come from task-level studies and enterprise
    surveys, not from effort estimation research. Applying them to full-project effort
    estimates involves uncertainty. The central estimate is the research midpoint;
    real impact depends on your codebase, team adoption rate, and task type."""),
]


# ── SECTION 7: GITHUB MODEL ───────────────────────────────────────────────────
story += [
    p("7. The GitHub-Trained Model", H1), rule(BLUE_BDR),
    p("""The COCOMO-81 model was trained on 1980s defense software. To show the
    methodology works on <b>modern software projects</b>, we collected 91 real
    GitHub repositories and trained the same FAHP + SVR pipeline on their data."""),
    sp(),

    p("7.1 Features", H2),
    data_table(
        ["Feature", "What it measures", "Why it matters"],
        [
            ["Contributors",       "Unique people who committed", "More people = more coordination overhead (Brooks's Law)"],
            ["PR Review Time",     "Avg days from PR open to merge", "Slow reviews = less capable / slower team"],
            ["PR Merge Rate",      "Fraction of PRs merged", "Low merge rate = rework and wasted effort"],
            ["Test File Ratio",    "Fraction of files that are tests", "High test coverage = mature engineering practices"],
            ["Language Count",     "Number of programming languages", "More languages = more context-switching overhead"],
            ["Commit Frequency",   "Commits per week", "Active development intensity signal"],
            ["Contributors × Freq","Interaction term", "Large fast-moving teams — captures combined scale"],
            ["AI Tool Adoption",   "Number of AI tools configured", "Copilot, Cursor, Claude Code, Agents (0–3)"],
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
    work — but it correlates with real project size and effort across the dataset."""),
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
    p("""The low MMRE therefore reflects a genuine team-size → effort signal that cannot be
    cleanly separated from shared measurement source. Jira-tracked hours as an independent
    ground truth would remove the overlap and give a clean accuracy figure."""),
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
    bullet("Re-fit scalers and FAHP weights entirely on training data"),
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

    p("8.1 Ablation Study — Does FAHP Actually Help?", H2),
    p("""A fair ablation must give the unweighted baseline its OWN hyperparameter search.
    Reusing the C/gamma/epsilon tuned for the FAHP-weighted feature space cripples the
    baseline and inflates the apparent benefit. Below, each variant is independently
    tuned over the same grid, with leakage-free nested evaluation:"""),
    data_table(
        ["Model variant", "Nested-CV MMRE", "FAHP comparison"],
        [
            ["Plain SVR (no FAHP, independently tuned)", f"{C_FAHP['unweighted_mmre']:.4f}", "baseline"],
            ["FAHP-Weighted SVR", f"{C_FAHP['weighted_mmre']:.4f}", f"{C_FAHP['effect_pct']:.1f}% lower MMRE in this comparison"],
        ],
        col_widths=[8*cm, 3*cm, 5.6*cm]
    ),
    p(f"""In matched nested validation, FAHP weighting changes COCOMO-81 MMRE by
    <b>{C_FAHP['effect_pct']:.1f}%</b> ({C_FAHP['unweighted_mmre']:.4f} to
    {C_FAHP['weighted_mmre']:.4f}). This is a sample-specific comparison, not proof that
    weighting improves every future dataset."""),
    sp(),
    p(f"""<b>GitHub comparison:</b> the independently tuned unweighted model scores
    {G_FAHP['unweighted_mmre']:.4f}, while the weighted model scores
    {G_FAHP['weighted_mmre']:.4f}, a {G_FAHP['effect_pct']:.1f}% lower MMRE in these
    nested folds. The result is reported as a comparison because it does not establish
    a universal causal improvement from FAHP."""),
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
    bullet("Log-transforms, MinMax-scales, and FAHP-weights the vector"),
    bullet("Passes it through the trained RBF kernel SVR"),
    bullet("Inverse-scales and expm1 transforms back to person-months"),
    p("""For these exact inputs the model returns <b>111 PM</b> — nearly identical to the
    COCOMO formula's 107 PM. This is expected: with a well-calibrated EAF (0.574) on a
    Semi-detached project squarely within the training distribution, the SVR and the
    COCOMO multiplicative formula converge. The SVR adds value on projects where
    non-linear interactions between drivers matter more — here, the drivers are
    moderate and the two methods agree."""),
    sp(),

    p("Step 4 — SHAP Decomposition (real model output, seed=42, representative background n=50)", H2),
    p("""Background: the 50 training projects with effort ≤ 600 PM (the 80th percentile).
    The 13 excluded projects are 1980s defence mega-programs (up to 11,400 PM) that skew the
    arithmetic mean to 620 PM — not representative of modern commercial software.
    The resulting base value is <b>115.8 PM</b>, close to the geometric mean (127 PM),
    and the SHAP values are in PM units throughout. Additivity verified: 0.00% error."""),
    sp(),
    data_table(
        ["Feature", "SHAP value", "Reading"],
        [
            ["LOC (40 KLOC)",  "+69.7 PM",  "40 KLOC is larger than the typical background project — main effort driver"],
            ["RELY (High)",    "+14.0 PM",  "Financial stakes push above baseline"],
            ["DevMode (Semi)", "+7.6 PM",   "Semi-detached slightly above the mixed background average"],
            ["TOOL (High)",    "−23.5 PM",  "Good tooling pulls effort down"],
            ["TIME (Nominal)", "−21.8 PM",  "No tight latency SLA — typical background has more pressure"],
            ["TURN (Low)",     "−15.1 PM",  "Fast CI/CD reduces friction"],
            ["PCAP (High)",    "−12.4 PM",  "Strong developers are faster than background average"],
            ["VEXP (High)",    "−11.3 PM",  "Cloud-experienced team"],
            ["ACAP (High)",    "−7.7 PM",   "Strong architects reduce effort"],
            ["MODP (High)",    "−4.6 PM",   "Good engineering practices"],
            ["All others",     "0.0 PM",    "DATA, CPLX, STOR, VIRT, AEXP, LEXP, SCED — negligible vs background"],
            ["Baseline",       "115.8 PM",  "Mean prediction over 50 representative training projects (seed=42, stable)"],
            ["<b>Total</b>",   "<b>110.8 PM</b>", "115.8 − 5.0 = 110.8  ·  additivity: 0.00% error"],
        ],
        col_widths=[3.5*cm, 2.5*cm, 10.6*cm]
    ),
    sp(),

    p("Step 5 — Schedule and Team", H2),
    formula_box("Calendar months = 2.5 × 111^0.38 = 2.5 × 5.98 ≈ 15 months"),
    formula_box("Recommended team = 111 / 15 ≈ 7 engineers"),
    sp(),

    p("Step 6 — GenAI Adjustment (Copilot adopted)", H2),
    formula_box("AI effort = 111 × 0.78 = 87 PM   (22% reduction)"),
    formula_box("AI calendar = 2.5 × 87^0.38 = 2.5 × 5.46 ≈ 14 months   |   AI team = 6 engineers"),
    formula_box(f"80% PI: 111 / (1 + {C_PI['q90']:+.3f}) = {111/(1+C_PI['q90']):.0f} PM  to  111 / (1 + {C_PI['q10']:+.3f}) = {111/(1+C_PI['q10']):.0f} PM"),
    callout(f"""<b>Summary for the FinTech API project (illustrative model output):</b><br/>
    SVR prediction: 110.8 PM  ·  COCOMO: 107 PM  ·  Models converge (well-calibrated inputs)<br/>
    Without AI: 111 PM, 15 months, 7 engineers<br/>
    With Copilot: 87 PM, 14 months, 6 engineers (saves ~24 PM = 22%)<br/>
    80% empirical interval: {111/(1+C_PI['q90']):.0f}–{111/(1+C_PI['q10']):.0f} PM (wide — reflects genuine estimation uncertainty)<br/>
    SHAP baseline: 115.8 PM (mean over 50 representative training projects, seed=42)<br/>
    Top effort driver: LOC (+69.7 PM); top reducers: TOOL (−23.5 PM), TIME (−21.8 PM)""",
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
    bullet("<b>Solver differences:</b> Sehra et al. (2018) used LSSVM; this implementation uses sklearn SVR. The weighted-kernel formulation is faithful; the solver is an acknowledged simplification."),
    bullet("<b>FAHP weight derivation:</b> Expert pairwise comparisons are replaced with Spearman correlation ratios — a data-driven proxy. In a full implementation, domain experts would provide the comparisons directly."),
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
    bullet("<b>Study variation:</b> The 22–55% range is wide. Task type, codebase complexity, and team adoption rate all moderate the effect. Use the central estimate for planning; use the range for risk scenarios."),
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
    bullet("Quantify how much AI tool adoption is actually saving your team (not a study estimate)"),
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
            ["Model training & validation",           "Month 3",   "FAHP-SVR model, SHAP integration, nested validation and chronological hold-out when dates permit"],
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
    callout("""<b>Next step:</b> Use the app's downloadable template and local validator.
    A feasibility review can assess completeness and target quality, but achievable MMRE is
    established only by leakage-free validation on the resulting dataset.""", GREEN_BG, GREEN),
    sp(12),
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
            ["Chang, D.Y. (1996). Applications of the extent analysis method on fuzzy AHP. European Journal of Operational Research, 95(3), 649–655.", "Extent analysis algorithm — FAHP weights"],
            ["Sehra, S.K., Brar, Y.S., Kaur, N. (2018). Multi-objective optimization for software effort estimation using FAHP and SVR. Soft Computing, 22(8).", "Core methodology this tool implements"],
            ["Peng, S. et al. (2023). The impact of AI coding assistants on developer productivity: A controlled experiment. GitHub Research.", "55% faster task completion in the controlled experiment"],
            ["McKinsey Global Institute (2023). The economic potential of generative AI.", "20–45% developer productivity gain"],
            ["Kalliamvakou, E. (2022). Research: Quantifying GitHub Copilot's impact on developer productivity. GitHub Blog.", "Copilot PR merge speed study"],
            ["Lundberg, S.M., Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.", "SHAP explainability framework"],
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
    subject="FAHP + Weighted-Kernel SVR + SHAP methodology",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
shutil.copyfile(OUT, PROJECT_OUT)

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
<h2>FAHP comparison</h2><table><thead><tr><th>Dataset</th><th>Unweighted MMRE</th><th>Weighted MMRE</th><th>Difference</th></tr></thead><tbody>
<tr><td>COCOMO-81</td><td>{C_FAHP['unweighted_mmre']:.4f}</td><td>{C_FAHP['weighted_mmre']:.4f}</td><td>{C_FAHP['effect_pct']:.1f}% lower</td></tr>
<tr><td>GitHub</td><td>{G_FAHP['unweighted_mmre']:.4f}</td><td>{G_FAHP['weighted_mmre']:.4f}</td><td>{G_FAHP['effect_pct']:.1f}% lower</td></tr>
</tbody></table><p>These are matched nested-validation comparisons, not a claim of universal improvement.</p>
<h2>Prediction interval</h2><p>Signed relative error is <code>(prediction - actual) / actual</code>. The empirical interval is <code>[prediction / (1 + q90), prediction / (1 + q10)]</code>, using pure outer-fold residuals.</p>
<h2>GitHub effort proxy limitation</h2><p class="note">At most {PROXY['commit_sample_limit']:,} commits are sampled per repository, approximately {PROXY['approximate_cap_author_months']:.0f} author-months. Predictions above that cap are extrapolations and have not been validated.</p>
<h2>Your Data</h2><p>No accuracy is promised before data is supplied. The app validates CSV/XLSX schemas and provides a template; training is intentionally disabled until completed-project outcomes and sufficient sample size are available.</p>
</body></html>"""
with open(HTML_OUT, "w", encoding="utf-8") as html_file:
    html_file.write(html_report)

print(f"PDF saved: {OUT}")
print(f"PDF saved: {PROJECT_OUT}")
print(f"HTML saved: {HTML_OUT}")
