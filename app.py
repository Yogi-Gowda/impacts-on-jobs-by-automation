import streamlit as st
import pandas as pd
import numpy as np
import joblib, os

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WorkScan · AI Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #f5f2ed !important;
    color: #1a1714;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    max-width: 1100px !important;
    padding: 0 2rem 5rem 2rem !important;
}

/* NAV */
.ws-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem 0 1.4rem 0;
    border-bottom: 1px solid #d8d2c8;
    margin-bottom: 3.5rem;
}
.ws-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem;
    color: #1a1714;
}
.ws-logo span { color: #c0392b; }
.ws-nav-right { display: flex; align-items: center; gap: 2rem; }
.ws-nav-links {
    display: flex; gap: 2rem;
    font-size: 0.8rem; font-weight: 500;
    color: #9a8f85; letter-spacing: 0.05em; text-transform: uppercase;
}
.ws-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; background: #1a1714;
    color: #f5f2ed; padding: 0.25rem 0.75rem;
    border-radius: 2px; letter-spacing: 0.1em;
}

/* HERO */
.ws-hero {
    display: grid;
    grid-template-columns: 55fr 45fr;
    gap: 4rem;
    align-items: center;
    margin-bottom: 4rem;
    padding-bottom: 3.5rem;
    border-bottom: 1px solid #d8d2c8;
}
.ws-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #c0392b; margin-bottom: 1rem;
}
.ws-hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: 4rem; line-height: 1.04;
    color: #1a1714; letter-spacing: -0.025em; margin-bottom: 1.4rem;
}
.ws-hero-title em { font-style: italic; color: #9a8f85; }
.ws-hero-desc {
    font-size: 0.93rem; line-height: 1.8; color: #5a504a;
    max-width: 380px; font-weight: 300;
}
.ws-stats { display: flex; flex-direction: column; }
.ws-stat-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.95rem 0; border-top: 1px solid #d8d2c8;
}
.ws-stat-row:last-child { border-bottom: 1px solid #d8d2c8; }
.ws-stat-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #9a8f85;
}
.ws-stat-val {
    font-family: 'Instrument Serif', serif; font-size: 2rem;
    color: #1a1714; letter-spacing: -0.02em;
}
.ws-stat-val.red { color: #c0392b; }

/* Section label */
.ws-section-label {
    font-family: 'DM Mono', monospace; font-size: 0.63rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #9a8f85;
    margin-bottom: 1.2rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid #d8d2c8;
}

/* Streamlit widget overrides */
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #7a6f65 !important;
    font-weight: 400 !important;
}
.stSelectbox > div > div {
    background: #fffefa !important;
    border-color: #cfc9bf !important;
    border-radius: 3px !important;
    color: #1a1714 !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #c0392b !important; box-shadow: none !important;
}
.stNumberInput > div > div > input {
    background: #fffefa !important; border-color: #cfc9bf !important;
    border-radius: 3px !important; color: #1a1714 !important;
    font-size: 0.9rem !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] { background: #1a1714 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: #c0392b !important; }

/* Button */
div.stButton > button {
    background: #1a1714 !important; color: #f5f2ed !important;
    border: none !important; border-radius: 3px !important;
    padding: 0.75rem 2.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    letter-spacing: 0.03em !important; cursor: pointer !important;
    transition: background 0.15s !important; width: auto !important;
}
div.stButton > button:hover { background: #c0392b !important; }

/* Derived cards */
.ws-derived-row {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 0.8rem; margin-bottom: 2.5rem;
}
.ws-derived-card {
    background: #fffefa; border: 1px solid #d8d2c8;
    border-radius: 3px; padding: 1rem 1.2rem;
}
.ws-derived-label {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #9a8f85; margin-bottom: 0.35rem;
}
.ws-derived-val {
    font-family: 'Instrument Serif', serif;
    font-size: 1.6rem; color: #1a1714; letter-spacing: -0.01em;
}
.ws-derived-val.pos { color: #27ae60; }
.ws-derived-val.neg { color: #c0392b; }

/* Result grid */
.ws-result-wrap { animation: fadeUp 0.35s ease forwards; margin-top: 0.5rem; }
@keyframes fadeUp {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0); }
}
.ws-result-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;
}
.ws-result-card {
    border: 1px solid #d8d2c8; border-radius: 3px;
    padding: 1.8rem 2rem; background: #fffefa;
    position: relative; overflow: hidden;
}
.ws-result-card::before {
    content: ""; position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
}
.ws-result-card.high::before    { background: #c0392b; }
.ws-result-card.moderate::before { background: #e67e22; }
.ws-result-card.low::before     { background: #27ae60; }

.ws-result-model {
    font-family: 'DM Mono', monospace; font-size: 0.63rem;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #9a8f85; margin-bottom: 0.8rem;
}
.ws-risk-big {
    font-family: 'Instrument Serif', serif; font-size: 2.6rem;
    letter-spacing: -0.02em; margin-bottom: 0.15rem;
}
.ws-risk-big.high     { color: #c0392b; }
.ws-risk-big.moderate { color: #e67e22; }
.ws-risk-big.low      { color: #27ae60; }
.ws-risk-sub { font-size: 0.8rem; color: #9a8f85; margin-bottom: 1.2rem; font-weight: 300; }

.ws-prob-row { margin-bottom: 0.5rem; }
.ws-prob-meta {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.06em; color: #9a8f85; margin-bottom: 0.25rem;
    text-transform: uppercase;
}
.ws-prob-track { height: 3px; background: #e8e2d8; border-radius: 2px; overflow: hidden; }
.ws-prob-fill  { height: 100%; border-radius: 2px; }
.fill-high     { background: #c0392b; }
.fill-moderate { background: #e67e22; }
.fill-low      { background: #27ae60; }

/* Insight */
.ws-insight {
    background: #1a1714; color: #e8e2d8; border-radius: 3px;
    padding: 1.3rem 1.6rem; font-size: 0.88rem;
    line-height: 1.72; font-weight: 300;
}
.ws-insight strong { color: #f5f2ed; font-weight: 600; }

/* Expander */
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #7a6f65 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] th {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important; color: #7a6f65 !important;
    background: #f5f2ed !important;
}
[data-testid="stDataFrame"] td {
    font-size: 0.84rem !important; color: #1a1714 !important;
}

/* Footer */
.ws-footer {
    margin-top: 5rem; padding-top: 1.4rem;
    border-top: 1px solid #d8d2c8;
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #b0a89e;
}

hr { border-color: #d8d2c8 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f2ed; }
::-webkit-scrollbar-thumb { background: #cfc9bf; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data & models
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    for p in ["ai_job_market2.csv", "/mnt/user-data/uploads/ai_job_market2.csv",
              "ai_job_market.csv",  "/mnt/user-data/uploads/ai_job_market.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_resource
def load_models():
    out = {}
    for name, fname in [("Random Forest", "random_forest_model.pkl"),
                         ("XGBoost",       "xgboost_model.pkl")]:
        if os.path.exists(fname):
            out[name] = joblib.load(fname)
    return out

df_raw = load_raw()
models  = load_models()

# Exact LabelEncoder mappings (alphabetical — matches sklearn fit on full column)
ENCODE = {
    "Industry":           {"Education":0,"Entertainment":1,"Finance":2,"Healthcare":3,
                           "IT":4,"Manufacturing":5,"Media":6,"Research":7,
                           "Retail":8,"Technology":9,"Transportation":10},
    "Job Status":         {"Active":0,"Declining":1,"Stable":2},
    "AI Impact Level":    {"High":0,"Low":1,"Moderate":2},
    "Required Education": {"Associate Degree":0,"Bachelor":1,"Doctorate":2,
                           "High School":3,"Master":4},
}
RISK_DECODE = {0:"High", 1:"Low", 2:"Moderate"}

INDUSTRIES = sorted(ENCODE["Industry"].keys())
STATUSES   = sorted(ENCODE["Job Status"].keys())
AI_IMPACTS = sorted(ENCODE["AI Impact Level"].keys())
EDUCATIONS = sorted(ENCODE["Required Education"].keys())
JOB_TITLES = sorted(df_raw["Job Title"].unique().tolist()) if df_raw is not None else []

RISK_DESC = {
    "High":     "This role faces significant displacement pressure from automation within the next decade. Core task sets are highly replicable by AI systems.",
    "Moderate": "Partial exposure — routine sub-tasks may be automated, but human judgment and domain expertise remain essential.",
    "Low":      "This role is resilient to automation with strong projected demand through 2030. Interpersonal, creative, or physical complexity offers protection.",
}

def build_features(d):
    return pd.DataFrame([{
        "Industry":                    ENCODE["Industry"].get(d["industry"], 0),
        "Job Status":                  ENCODE["Job Status"].get(d["job_status"], 0),
        "AI Impact Level":             ENCODE["AI Impact Level"].get(d["ai_impact"], 0),
        "Median Salary (USD)":         d["salary"],
        "Required Education":          ENCODE["Required Education"].get(d["education"], 0),
        "Experience Required (Years)": d["experience"],
        "Job Openings (2024)":         d["openings_24"],
        "Projected Openings (2030)":   d["openings_30"],
        "Remote Work Ratio (%)":       d["remote"],
        "Gender Diversity (%)":        d["diversity"],
        "Growth_Rate":                 d["openings_30"] - d["openings_24"],
        "Salary_Experience_Ratio":     d["salary"] / (d["experience"] + 1),
        "Remote_Impact":               d["remote"] * d["diversity"],
    }])


# ─────────────────────────────────────────────────────────────────────────────
# NAV
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ws-nav">
    <div class="ws-logo">Work<span>Scan</span></div>
    <div class="ws-nav-right">
        <div class="ws-nav-links">
            <span>Analyzer</span>
            <span>Dataset</span>
        </div>
        <div class="ws-badge">Beta v2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
n_rec   = f"{len(df_raw):,}"                                   if df_raw is not None else "30,000"
avg_sal = f"${df_raw['Median Salary (USD)'].mean():,.0f}"       if df_raw is not None else "—"
avg_rsk = f"{df_raw['Automation Risk (%)'].mean():.1f}%"        if df_raw is not None else "—"
n_ttl   = str(df_raw["Job Title"].nunique())                    if df_raw is not None else "525"

st.markdown(f"""
<div class="ws-hero">
  <div>
    <div class="ws-eyebrow">◈ AI Labor Intelligence Platform</div>
    <div class="ws-hero-title">How safe is<br>your <em>job</em><br>from AI?</div>
    <div class="ws-hero-desc">
      WorkScan uses Random Forest &amp; XGBoost models trained on 30,000 real
      job records to predict your role's automation risk — giving you a
      data-backed answer, not a guess.
    </div>
  </div>
  <div class="ws-stats">
    <div class="ws-stat-row">
      <span class="ws-stat-label">Records analyzed</span>
      <span class="ws-stat-val">{n_rec}</span>
    </div>
    <div class="ws-stat-row">
      <span class="ws-stat-label">Unique job titles</span>
      <span class="ws-stat-val">{n_ttl}</span>
    </div>
    <div class="ws-stat-row">
      <span class="ws-stat-label">Average salary</span>
      <span class="ws-stat-val">{avg_sal}</span>
    </div>
    <div class="ws-stat-row">
      <span class="ws-stat-label">Avg. automation risk</span>
      <span class="ws-stat-val red">{avg_rsk}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ws-section-label">01 — Role Identity</div>', unsafe_allow_html=True)

job_title = st.selectbox(
    "Job Title",
    ["— select a title —"] + JOB_TITLES,
)

# Auto-fill defaults when a title is picked
af = None
if job_title != "— select a title —" and df_raw is not None:
    rows = df_raw[df_raw["Job Title"] == job_title]
    if not rows.empty:
        af = rows.iloc[0]

def safe_idx(lst, val, default=0):
    try: return lst.index(val)
    except: return default

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    industry = st.selectbox("Industry", INDUSTRIES,
        index=safe_idx(INDUSTRIES, af["Industry"]) if af is not None else 0)
with c2:
    job_status = st.selectbox("Job Status", STATUSES,
        index=safe_idx(STATUSES, af["Job Status"]) if af is not None else 0)
with c3:
    ai_impact = st.selectbox("AI Impact Level", AI_IMPACTS,
        index=safe_idx(AI_IMPACTS, af["AI Impact Level"]) if af is not None else 0)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4:
    education = st.selectbox("Required Education", EDUCATIONS,
        index=safe_idx(EDUCATIONS, af["Required Education"]) if af is not None else 1)
with c5:
    experience = st.slider("Experience Required (Years)", 0, 20,
        int(af["Experience Required (Years)"]) if af is not None else 3)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ws-section-label">02 — Market Signals</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    salary = st.number_input("Median Salary (USD)", 25_000, 300_000,
        int(af["Median Salary (USD)"]) if af is not None else 75_000, step=1_000)
with m2:
    openings_24 = st.number_input("Job Openings (2024)", 0, 150_000,
        int(af["Job Openings (2024)"]) if af is not None else 5_000, step=100)
with m3:
    openings_30 = st.number_input("Projected Openings (2030)", 0, 150_000,
        int(af["Projected Openings (2030)"]) if af is not None else 6_000, step=100)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
r1, r2 = st.columns(2)
with r1:
    remote = st.slider("Remote Work Ratio (%)", 0, 100,
        int(af["Remote Work Ratio (%)"]) if af is not None else 40)
with r2:
    diversity = st.slider("Gender Diversity (%)", 0, 100,
        int(af["Gender Diversity (%)"]) if af is not None else 45)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ws-section-label">03 — Run Analysis</div>', unsafe_allow_html=True)

btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1, 2])
with btn_c1:
    model_choice = st.selectbox("Model", ["Both Models", "XGBoost", "Random Forest"])
with btn_c2:
    st.markdown("<div style='height:1.72rem'></div>", unsafe_allow_html=True)
    run = st.button("Analyze →")


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED FEATURES PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
growth      = openings_30 - openings_24
sal_ratio   = salary / (experience + 1)
r_impact    = remote * diversity
growth_cls  = "pos" if growth >= 0 else "neg"
growth_sign = "+" if growth >= 0 else ""
growth_fmt  = f"{growth_sign}{growth:,}"
sal_fmt     = f"${sal_ratio:,.0f}"
rim_fmt     = f"{r_impact:,.0f}"

st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ws-section-label">Computed Features (live)</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="ws-derived-row">
  <div class="ws-derived-card">
    <div class="ws-derived-label">Job Growth Rate</div>
    <div class="ws-derived-val {growth_cls}">{growth_fmt}</div>
  </div>
  <div class="ws-derived-card">
    <div class="ws-derived-label">Salary / Experience</div>
    <div class="ws-derived-val">{sal_fmt}</div>
  </div>
  <div class="ws-derived-card">
    <div class="ws-derived-label">Remote × Diversity</div>
    <div class="ws-derived-val">{rim_fmt}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if run:
    if not models:
        st.error("⚠️  No model files found. Run `prep.ipynb` to generate `random_forest_model.pkl` and `xgboost_model.pkl`, then place them in the same directory as this file.")
    else:
        d = dict(industry=industry, job_status=job_status, ai_impact=ai_impact,
                 education=education, salary=salary, experience=experience,
                 openings_24=openings_24, openings_30=openings_30,
                 remote=remote, diversity=diversity)
        X = build_features(d)

        sel = (list(models.items())         if model_choice == "Both Models"
               else [(model_choice, models[model_choice])] if model_choice in models
               else [])

        if not sel:
            st.warning(f"'{model_choice}' model file not loaded.")
        else:
            st.markdown('<div class="ws-section-label" style="margin-top:2rem">Prediction Results</div>',
                        unsafe_allow_html=True)

            title_lbl = job_title if job_title != "— select a title —" else "Selected Role"

            risks = []

            # Open the result grid
            st.markdown('<div class="ws-result-wrap"><div class="ws-result-grid">', unsafe_allow_html=True)

            cols = st.columns(len(sel))
            for col, (name, model) in zip(cols, sel):
                enc  = int(model.predict(X)[0])
                risk = RISK_DECODE.get(enc, str(enc))
                risks.append(risk)
                cls  = risk.lower()

                # Build probability bars HTML
                pb = ""
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    for idx, lbl in enumerate(["High", "Low", "Moderate"]):
                        p     = probs[idx] * 100
                        lcls  = lbl.lower()
                        pb += (
                            f'<div class="ws-prob-row">'
                            f'<div class="ws-prob-meta"><span>{lbl}</span><span>{p:.1f}%</span></div>'
                            f'<div class="ws-prob-track">'
                            f'<div class="ws-prob-fill fill-{lcls}" style="width:{p:.1f}%"></div>'
                            f'</div></div>'
                        )

                card = (
                    f'<div class="ws-result-card {cls}">'
                    f'<div class="ws-result-model">{name}</div>'
                    f'<div class="ws-risk-big {cls}">{risk} Risk</div>'
                    f'<div class="ws-risk-sub">{title_lbl}</div>'
                    f'{pb}'
                    f'</div>'
                )
                with col:
                    st.markdown(card, unsafe_allow_html=True)

            # Close the grid
            st.markdown('</div>', unsafe_allow_html=True)

            # Insight bar
            consensus = risks[0] if len(set(risks)) == 1 else "Mixed"
            if consensus in RISK_DESC:
                insight = f"<strong>{consensus} Risk —</strong> {RISK_DESC[consensus]}"
            else:
                insight = "<strong>Models disagree.</strong> Random Forest and XGBoost returned different classifications. Review the probability distributions above for a fuller picture."

            st.markdown(
                f'<div class="ws-insight">{insight}</div></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
if df_raw is not None:
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ws-section-label">Dataset Explorer</div>', unsafe_allow_html=True)

    with st.expander("Browse records & summaries"):
        t1, t2, t3 = st.tabs(["Raw Data", "Industry Summary", "Risk Distribution"])

        with t1:
            st.dataframe(df_raw.head(100), use_container_width=True, hide_index=True)

        with t2:
            summ = (df_raw.groupby("Industry")
                .agg(Jobs=("Job Title","count"),
                     Avg_Salary=("Median Salary (USD)","mean"),
                     Avg_Risk=("Automation Risk (%)","mean"))
                .reset_index()
                .sort_values("Avg_Risk", ascending=False)
                .rename(columns={"Avg_Salary":"Avg Salary (USD)","Avg_Risk":"Avg Auto Risk (%)"}))
            summ["Avg Salary (USD)"]  = summ["Avg Salary (USD)"].round(0).astype(int)
            summ["Avg Auto Risk (%)"] = summ["Avg Auto Risk (%)"].round(1)
            st.dataframe(summ, use_container_width=True, hide_index=True)

        with t3:
            def bucket(x):
                if x >= 70: return "High  (≥70%)"
                if x >= 40: return "Moderate  (40–69%)"
                return "Low  (<40%)"
            rd = df_raw["Automation Risk (%)"].apply(bucket).value_counts().reset_index()
            rd.columns = ["Risk Level", "Count"]
            rd["Share (%)"] = (rd["Count"] / rd["Count"].sum() * 100).round(1)
            st.dataframe(rd, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
loaded = ", ".join(models.keys()) if models else "No models loaded — run prep.ipynb first"
st.markdown(f"""
<div class="ws-footer">
  <span>WorkScan · AI Risk Intelligence</span>
  <span>Models: {loaded}</span>
  <span>30,000 records · 525 job titles · 13 features</span>
</div>
""", unsafe_allow_html=True)

print("Yogesh")