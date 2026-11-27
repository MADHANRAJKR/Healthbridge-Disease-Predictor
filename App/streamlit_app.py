"""
HealthBridge — Streamlit Symptom Checker UI
============================================
Run:
    streamlit run App/streamlit_app.py
    (from repo root, with the FastAPI server running on port 8000)
"""

import base64
import sys
from pathlib import Path

import httpx
import streamlit as st
import pandas as pd
import altair as alt

# Allow fallback: if API is down, call model directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthBridge · AI Symptom Checker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-blue medical theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Root palette ---- */
:root {
    --bg-dark:       #0d1117;
    --bg-card:       #161b22;
    --bg-glass:      rgba(22, 27, 34, 0.85);
    --accent:        #2563eb;
    --accent-light:  #3b82f6;
    --accent-glow:   rgba(37, 99, 235, 0.25);
    --success:       #10b981;
    --warning:       #f59e0b;
    --danger:        #ef4444;
    --text-primary:  #f0f6fc;
    --text-secondary:#8b949e;
    --border:        rgba(48, 54, 61, 0.9);
    --radius:        12px;
}

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f172a 50%, #0d1117 100%);
    color: var(--text-primary);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent-light);
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ---- Metric cards ---- */
.metric-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px var(--accent-glow);
}
.metric-label { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { color: var(--text-primary); font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
.metric-sub   { color: var(--accent-light); font-size: 0.85rem; }

/* ---- Disease badge ---- */
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    color: white;
    padding: 0.5rem 1.4rem;
    border-radius: 999px;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 16px var(--accent-glow);
}

/* ---- Confidence bar ---- */
.conf-track { background: rgba(255,255,255,0.08); border-radius: 999px; height: 10px; margin-top: 0.4rem; }
.conf-fill  { height: 10px; border-radius: 999px;
              background: linear-gradient(90deg, var(--accent), var(--success)); }

/* ---- Section header ---- */
.section-header {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-secondary); margin: 1.5rem 0 0.6rem;
    border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;
}

/* ---- Disclaimer ---- */
.disclaimer {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: var(--radius);
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    font-size: 0.8rem;
    margin-top: 1rem;
}

/* ---- Predict button ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--accent-glow) !important;
}

/* ---- Slider labels ---- */
.stSlider label { color: var(--text-primary) !important; font-weight: 500; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown('<div style="font-size:3rem;padding-top:0.3rem">🏥</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style="margin:0;font-size:2rem;font-weight:700;
               background:linear-gradient(135deg,#60a5fa,#a78bfa);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        HealthBridge · AI Symptom Checker
    </h1>
    <p style="margin:0;color:#8b949e;font-size:0.9rem">
        Powered by Machine Learning · SHAP Explainability · 30 Disease Classes
    </p>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — symptom inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Patient Symptoms")

    st.markdown("**Demographics**")
    age    = st.slider("Age", 1, 99, 35, key="age")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True, key="gender")

    st.markdown("---")
    st.markdown("**Vital Signs**")
    bmi          = st.slider("BMI", 10.0, 60.0, 22.0, 0.1, key="bmi")
    bp_sys       = st.slider("BP Systolic (mmHg)", 60, 220, 120, key="bp_sys")
    bp_dia       = st.slider("BP Diastolic (mmHg)", 40, 140, 80, key="bp_dia")
    spo2         = st.slider("SpO₂ (%)", 70, 100, 98, key="spo2")
    glucose      = st.slider("Blood Glucose (mg/dL)", 40, 500, 90, key="glucose")
    hba1c        = st.slider("HbA1c (%)", 3.5, 15.0, 5.0, 0.1, key="hba1c")

    st.markdown("---")
    st.markdown("**Symptoms Present**")
    fever          = st.checkbox("🌡️ Fever",           key="fever")
    cough          = st.checkbox("😮‍💨 Cough",           key="cough")
    fatigue        = st.checkbox("😴 Fatigue",          key="fatigue")
    breathlessness = st.checkbox("💨 Breathlessness",   key="breathlessness")
    chest_pain     = st.checkbox("💔 Chest Pain",       key="chest_pain")
    nausea         = st.checkbox("🤢 Nausea",           key="nausea")
    headache       = st.checkbox("🤕 Headache",         key="headache")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Disease", use_container_width=True)

# ---------------------------------------------------------------------------
# Build payload
# ---------------------------------------------------------------------------
payload = {
    "Age":            age,
    "Gender":         1 if gender == "Male" else 0,
    "Fever":          int(fever),
    "Cough":          int(cough),
    "Fatigue":        int(fatigue),
    "Breathlessness": int(breathlessness),
    "Chest_Pain":     int(chest_pain),
    "Nausea":         int(nausea),
    "Headache":       int(headache),
    "BMI":            bmi,
    "BP_Systolic":    bp_sys,
    "BP_Diastolic":   bp_dia,
    "SpO2":           spo2,
    "Glucose_mg_dL":  glucose,
    "HbA1c":          hba1c,
}

# ---------------------------------------------------------------------------
# Main area — empty state / results
# ---------------------------------------------------------------------------
if not predict_btn:
    # Welcome card
    st.markdown("""
    <div class="metric-card" style="margin-top:2rem;text-align:left;padding:2rem">
        <h2 style="color:#60a5fa;margin:0 0 0.8rem">How to use</h2>
        <ol style="color:#8b949e;line-height:1.9;padding-left:1.2rem">
            <li>Fill in the patient's <b style="color:#f0f6fc">demographics</b> and <b style="color:#f0f6fc">vital signs</b> in the sidebar.</li>
            <li>Check any <b style="color:#f0f6fc">symptoms</b> that are present.</li>
            <li>Click <b style="color:#3b82f6">🔍 Predict Disease</b>.</li>
            <li>Review the prediction, confidence score, top-3 candidates, and SHAP explanation.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown('<p class="section-header">System Overview</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="metric-card"><div class="metric-label">Disease Classes</div><div class="metric-value">30</div><div class="metric-sub">covered conditions</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">&gt;80%</div><div class="metric-sub">on held-out test set</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="metric-card"><div class="metric-label">Input Features</div><div class="metric-value">15</div><div class="metric-sub">symptoms + vitals</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="metric-card"><div class="metric-label">Explainability</div><div class="metric-value">SHAP</div><div class="metric-sub">per-prediction</div></div>', unsafe_allow_html=True)

else:
    # ---- Call API ----
    with st.spinner("Analysing symptoms …"):
        try:
            resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            api_ok = True
        except Exception as exc:
            api_ok = False
            err_msg = str(exc)

    if not api_ok:
        st.error(f"⚠️ Could not reach the API at `{API_URL}/predict`. "
                 f"Make sure the FastAPI server is running.\n\n`{err_msg}`")
    else:
        pred      = data["predicted_disease"]
        conf      = data["confidence"]
        top3      = data["top_3"]
        shap5     = data["shap_top5"]
        chart_b64 = data.get("shap_chart_b64")

        # ---- Top result ----
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns([3, 2, 2])

        with r1:
            conf_pct = int(conf * 100)
            conf_color = "#10b981" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 50 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="text-align:left">
                <div class="metric-label">Most Likely Diagnosis</div>
                <div style="margin:0.8rem 0">
                    <span class="disease-badge">{pred}</span>
                </div>
                <div class="metric-label" style="margin-top:0.8rem">Model Confidence</div>
                <div style="font-size:1.5rem;font-weight:700;color:{conf_color}">{conf_pct}%</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{conf_color},{conf_color}99)"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Top-3 bar chart via Altair
            df_top3 = pd.DataFrame([
                {"Disease": d["disease"], "Probability": round(d["probability"] * 100, 1)}
                for d in top3
            ])
            chart = (
                alt.Chart(df_top3)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Disease:N", sort="-y", axis=alt.Axis(labelAngle=-15)),
                    y=alt.Y("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Disease:N", scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["Disease", "Probability"],
                )
                .properties(title="Top-3 Predictions", height=200)
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False, labelColor="#8b949e", titleColor="#8b949e")
                .configure_title(color="#f0f6fc", fontSize=13)
            )
            st.altair_chart(chart, use_container_width=True)

        with r3:
            # SHAP top-5 feature table
            st.markdown('<div class="metric-label" style="margin-bottom:0.5rem">Top SHAP Features</div>', unsafe_allow_html=True)
            df_shap = pd.DataFrame([
                {"Feature": f["feature"], "Impact": round(f["shap_value"], 4)}
                for f in shap5
            ])
            st.dataframe(
                df_shap,
                use_container_width=True,
                hide_index=True,
                height=200,
            )

        # ---- SHAP chart ----
        if chart_b64:
            st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
            img_bytes = base64.b64decode(chart_b64)
            st.image(img_bytes, caption="Feature contributions to this prediction (red = pushes toward predicted class, blue = pushes away)", use_column_width=True)

        # ---- Raw vitals summary ----
        with st.expander("📋 Submitted Patient Data"):
            df_in = pd.DataFrame([payload])
            st.dataframe(df_in, use_container_width=True, hide_index=True)

        # ---- Disclaimer ----
        st.markdown(f"""
        <div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b> {data.get('disclaimer',
            'This is an AI prototype for educational purposes. It is NOT a clinical diagnosis.')}
        </div>
        """, unsafe_allow_html=True)
