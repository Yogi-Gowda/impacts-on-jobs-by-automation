import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# ── Page config 
st.set_page_config(
    page_title="AI Job Risk Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c8c8e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7070a0 !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d1f 0%, #12122a 50%, #0a0a18 100%);
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, #4040ff22 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5555cc;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: #6666ff; }
.hero-sub {
    font-size: 1rem;
    color: #8888aa;
    max-width: 520px;
    line-height: 1.6;
}

/* ── Cards ── */
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3333aa; }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5555aa;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-value.accent { color: #6666ff; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.6rem;
    border-radius: 40px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.risk-High   { background: #ff333322; border: 1px solid #ff3333; color: #ff6666; }
.risk-Moderate { background: #ff880022; border: 1px solid #ff8800; color: #ffaa44; }
.risk-Low    { background: #00cc6622; border: 1px solid #00cc66; color: #00ee88; }

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0f0f1a, #12122a);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5555aa;
    margin-bottom: 0.8rem;
}

/* ── Section headings ── */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5555aa;
    border-left: 3px solid #4444cc;
    padding-left: 0.7rem;
    margin: 2rem 0 1rem 0;
}

/* ── Predict button ── */
div.stButton > button {
    background: #3333aa;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
}
div.stButton > button:hover {
    background: #5555cc;
    transform: translateY(-1px);
}

/* ── Data table ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e30;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #12121f !important;
    border-color: #2a2a4a !important;
    color: #e0e0f0 !important;
    border-radius: 8px !important;
}

/* ── Info box ── */
.info-box {
    background: #12122a;
    border: 1px solid #2a2a5a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #9090c0;
    line-height: 1.6;
}
.info-box code {
    font-family: 'Space Mono', monospace;
    color: #7777ff;
    background: #1a1a3a;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.8rem;
}

/* ── Divider ── */
hr { border-color: #1e1e30 !important; }

/* Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers 
INDUSTRY_OPTIONS = [
    "Technology", "Healthcare", "Finance", "Education", "Manufacturing",
    "Retail", "Transportation", "Entertainment", "Energy", "Construction"
]
JOB_STATUS_OPTIONS   = ["Active", "Declining", "Stable"]
AI_IMPACT_OPTIONS    = ["Low", "Moderate", "High"]
EDUCATION_OPTIONS    = ["High School", "Bachelor", "Master", "PhD"]

def encode_inputs(data: dict) -> pd.DataFrame:
    """Replicate the LabelEncoder logic from prep.ipynb (alphabetical order)."""
    mappings = {
        "Industry":            {v: i for i, v in enumerate(sorted(INDUSTRY_OPTIONS))},
        "Job Status":          {v: i for i, v in enumerate(sorted(JOB_STATUS_OPTIONS))},
        "AI Impact Level":     {v: i for i, v in enumerate(sorted(AI_IMPACT_OPTIONS))},
        "Required Education":  {v: i for i, v in enumerate(sorted(EDUCATION_OPTIONS))},
    }
    row = {
        "Industry":                    mappings["Industry"].get(data["Industry"], 0),
        "Job Status":                  mappings["Job Status"].get(data["Job Status"], 0),
        "AI Impact Level":             mappings["AI Impact Level"].get(data["AI Impact Level"], 0),
        "Median Salary (USD)":         data["Median Salary (USD)"],
        "Required Education":          mappings["Required Education"].get(data["Required Education"], 0),
        "Experience Required (Years)": data["Experience Required (Years)"],
        "Job Openings (2024)":         data["Job Openings (2024)"],
        "Projected Openings (2030)":   data["Projected Openings (2030)"],
        "Remote Work Ratio (%)":       data["Remote Work Ratio (%)"],
        "Gender Diversity (%)":        data["Gender Diversity (%)"],
        "Growth_Rate":                 data["Projected Openings (2030)"] - data["Job Openings (2024)"],
        "Salary_Experience_Ratio":     data["Median Salary (USD)"] / (data["Experience Required (Years)"] + 1),
        "Remote_Impact":               data["Remote Work Ratio (%)"] * data["Gender Diversity (%)"],
    }
    return pd.DataFrame([row])

def risk_color(label):
    colors = {"High": "#ff4444", "Moderate": "#ff9900", "Low": "#00dd77"}
    return colors.get(label, "#aaaaaa")

def load_models():
    models = {}
    for name, path in [("Random Forest", "random_forest_model.pkl"),
                        ("XGBoost",       "xgboost_model.pkl")]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

RISK_DECODE = {0: "High", 1: "Low", 2: "Moderate"}


# ── Hero 
st.markdown("""
<div class="hero">
    <div class="hero-tag">⬡ AI Labor Intelligence</div>
    <div class="hero-title">Job Automation<br><span>Risk Analyzer</span></div>
    <div class="hero-sub">
        Predict the automation risk level of any role using Random Forest &amp;
        XGBoost models trained on 15,000 real job records.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Dataset stats 
@st.cache_data
def load_data():
    path = "/mnt/user-data/uploads/ai_job_market.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_raw = load_data()

if df_raw is not None:
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("Total Records",   f"{len(df_raw):,}",              "accent"),
        ("Industries",      df_raw["Industry"].nunique(),     ""),
        ("Avg Salary (USD)",f"${df_raw['Median Salary (USD)'].mean():,.0f}", "accent"),
        ("Avg Auto Risk",   f"{df_raw['Automation Risk (%)'].mean():.1f}%", ""),
    ]
    for col, (label, value, cls) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)


# ── Sidebar inputs 
with st.sidebar:
    st.markdown("### 🔬 Job Parameters")
    st.markdown("---")

    industry        = st.selectbox("Industry",              INDUSTRY_OPTIONS)
    job_status      = st.selectbox("Job Status",            JOB_STATUS_OPTIONS)
    ai_impact       = st.selectbox("AI Impact Level",       AI_IMPACT_OPTIONS)
    education       = st.selectbox("Required Education",    EDUCATION_OPTIONS)

    st.markdown("---")
    salary          = st.number_input("Median Salary (USD)",         10_000, 300_000, 75_000, step=1_000)
    experience      = st.slider("Experience Required (Years)",        0, 20, 3)
    openings_2024   = st.number_input("Job Openings (2024)",          0, 100_000, 5_000, step=100)
    openings_2030   = st.number_input("Projected Openings (2030)",    0, 100_000, 6_000, step=100)
    remote_ratio    = st.slider("Remote Work Ratio (%)",              0, 100, 40)
    gender_div      = st.slider("Gender Diversity (%)",               0, 100, 45)

    st.markdown("---")
    model_choice = st.selectbox("Model", ["XGBoost", "Random Forest", "Both"])
    predict_btn  = st.button("⚡  Predict Risk Level")


# ── Derived features preview 
growth_rate = openings_2030 - openings_2024
sal_exp_ratio = salary / (experience + 1)
remote_impact = remote_ratio * gender_div

st.markdown('<div class="section-head">Derived Features</div>', unsafe_allow_html=True)
dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Growth Rate</div>
        <div class="metric-value" style="font-size:1.5rem;color:{'#00ee88' if growth_rate>=0 else '#ff6666'}">
            {'+' if growth_rate>=0 else ''}{growth_rate:,}
        </div>
    </div>""", unsafe_allow_html=True)
with dc2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Salary / Exp Ratio</div>
        <div class="metric-value" style="font-size:1.5rem;">${sal_exp_ratio:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with dc3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Remote Impact Score</div>
        <div class="metric-value" style="font-size:1.5rem;">{remote_impact:,}</div>
    </div>""", unsafe_allow_html=True)


# ── Prediction 
if predict_btn:
    input_data = {
        "Industry":                    industry,
        "Job Status":                  job_status,
        "AI Impact Level":             ai_impact,
        "Median Salary (USD)":         salary,
        "Required Education":          education,
        "Experience Required (Years)": experience,
        "Job Openings (2024)":         openings_2024,
        "Projected Openings (2030)":   openings_2030,
        "Remote Work Ratio (%)":       remote_ratio,
        "Gender Diversity (%)":        gender_div,
    }
    X_input = encode_inputs(input_data)

    models = load_models()

    if not models:
        st.error("⚠️ No model files found. Please place `random_forest_model.pkl` and `xgboost_model.pkl` in the working directory.")
    else:
        st.markdown('<div class="section-head">Prediction Results</div>', unsafe_allow_html=True)

        selected = (
            list(models.items()) if model_choice == "Both"
            else [(model_choice, models[model_choice])] if model_choice in models
            else []
        )

        if not selected:
            st.warning(f"Model '{model_choice}' not loaded. Check that the .pkl file exists.")
        else:
            res_cols = st.columns(len(selected))
            for col, (name, model) in zip(res_cols, selected):
                pred_encoded = model.predict(X_input)[0]
                risk_label   = RISK_DECODE.get(int(pred_encoded), str(pred_encoded))

                # Probability bar (if supported)
                proba_html = ""
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    for idx, cls in enumerate(["High", "Low", "Moderate"]):
                        p = proba[idx] * 100
                        bar_color = risk_color(cls)
                        proba_html += f"""
                        <div style="margin-bottom:0.5rem;">
                            <div style="display:flex;justify-content:space-between;
                                font-family:'Space Mono',monospace;font-size:0.7rem;
                                color:#6666aa;margin-bottom:3px;">
                                <span>{cls}</span><span>{p:.1f}%</span>
                            </div>
                            <div style="background:#1a1a2e;border-radius:4px;height:6px;overflow:hidden;">
                                <div style="width:{p:.1f}%;height:100%;
                                    background:{bar_color};border-radius:4px;
                                    transition:width 0.6s ease;"></div>
                            </div>
                        </div>"""

                with col:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">{name}</div>
                        <div style="margin:1rem 0;">
                            <span class="risk-badge risk-{risk_label}">
                                {'🔴' if risk_label=='High' else '🟡' if risk_label=='Moderate' else '🟢'}
                                &nbsp;{risk_label} Risk
                            </span>
                        </div>
                        {f'<div style="margin-top:1.2rem">{proba_html}</div>' if proba_html else ''}
                    </div>""", unsafe_allow_html=True)


# ── Data Explorer 
if df_raw is not None:
    st.markdown('<div class="section-head">Dataset Explorer</div>', unsafe_allow_html=True)

    with st.expander("Browse raw data & industry breakdown", expanded=False):
        tab1, tab2 = st.tabs(["📋 Raw Data", "📊 Industry Stats"])

        with tab1:
            st.dataframe(
                df_raw.head(50),
                use_container_width=True,
                hide_index=True,
            )

        with tab2:
            summary = (
                df_raw.groupby("Industry")
                .agg(
                    Count=("Job Title", "count"),
                    Avg_Salary=("Median Salary (USD)", "mean"),
                    Avg_Risk=("Automation Risk (%)", "mean"),
                )
                .reset_index()
                .sort_values("Avg_Risk", ascending=False)
                .rename(columns={
                    "Avg_Salary": "Avg Salary (USD)",
                    "Avg_Risk":   "Avg Auto Risk (%)",
                })
            )
            summary["Avg Salary (USD)"]  = summary["Avg Salary (USD)"].round(0).astype(int)
            summary["Avg Auto Risk (%)"] = summary["Avg Auto Risk (%)"].round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ── Footer 
st.markdown("---")
st.markdown("""
<div class="info-box">
    <strong>How to run:</strong> Place <code>random_forest_model.pkl</code> and
    <code>xgboost_model.pkl</code> (generated by <code>prep.ipynb</code>) in the
    same directory as this file, then run <code>streamlit run app.py</code>.
    The models are trained to classify jobs as <code>High</code>, <code>Moderate</code>,
    or <code>Low</code> automation risk using 13 engineered features.
</div>
""", unsafe_allow_html=True)