# -*- coding: utf-8 -*-
"""app.py —— Gastric Cancer Intraoperative Plasma Transfusion Risk Prediction (English UI)"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Gastric Cancer Plasma Transfusion Prediction",
    page_icon="🩺",
    layout="wide",
)

# ============ Paths ============
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    pre   = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    return model, pre

model, pre = load_artifacts()
CONT   = pre['continuous_vars']
BIN    = pre['binary_vars']
MED    = pre['medians']
MOD    = pre['modes']
THR    = pre['youden_threshold']
OPMAP  = pre['operative_time_map']
SURMAP = pre['surgery_type_map']

st.title("🩺 Gastric Cancer Intraoperative Plasma Transfusion Risk Prediction")
st.caption("XGBoost-based 11-variable machine learning model · For research use only, not a substitute for clinical judgment")
st.markdown(f"**Decision threshold (Youden)**：`{THR:.4f}` — probability ≥ this value is classified as **High Risk**")

with st.form("form"):
    st.subheader("① Continuous Variables")
    c1, c2, c3, c4 = st.columns(4)
    age   = c1.number_input("Age (years)",           0.0, 120.0,  float(MED.get('Age', 60)),   1.0)
    hb    = c2.number_input("Hemoglobin Hb (g/L)",   0.0, 250.0,  float(MED.get('Hb', 120)),   1.0)
    ast   = c3.number_input("AST (U/L)",             0.0, 2000.0, float(MED.get('AST', 25)),  1.0)
    egfr  = c4.number_input("eGFR (mL/min/1.73m²)",  0.0, 200.0,  float(MED.get('eGFR', 90)),  1.0)
    c5, c6, c7, _ = st.columns(4)
    diam  = c5.number_input("Minimum Tumor Diameter (cm)", 0.0, 30.0,   float(MED.get('Min_Diameter', 3)), 0.1)
    nrs   = c6.number_input("NRS2002 Score",               0.0, 7.0,    float(MED.get('NRS2002', 2)),      1.0)
    ca125 = c7.number_input("CA125 (U/mL)",                0.0, 5000.0, float(MED.get('CA125', 15)),       1.0)

    st.subheader("② Categorical Variables")
    c8, c9, c10, c11 = st.columns(4)
    op_time  = c8.selectbox("Operative Time", list(OPMAP.keys()))
    surgery  = c9.selectbox("Surgery Type", ["Elective", "Urgent"])
    htn      = c10.selectbox("Hypertension (HTN)", ["No", "Yes"])
    anticoag = c11.selectbox("Anticoagulation Use", ["No", "Yes"])

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    row = {
        'Age': age, 'Hb': hb, 'AST': ast, 'eGFR': egfr,
        'Min_Diameter': diam, 'NRS2002': nrs, 'CA125': ca125,
        'Operative_Time': OPMAP[op_time],
        'HTN':            1 if htn.startswith("Yes") else 0,
        'Anticoagulation': 1 if anticoag.startswith("Yes") else 0,
        'Surgery_Type':   1 if surgery.startswith("Elective") else 0,
    }
    X_in = pd.DataFrame([[row[f] for f in pre['features']]], columns=pre['features'])
    prob = float(model.predict_proba(X_in.values)[0, 1])

    st.divider()
    st.subheader("📊 Prediction Result")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Predicted Probability of Plasma Transfusion", f"{prob:.1%}")
    with col_b:
        if prob >= THR:
            st.error(f"🔴 **High Risk Group**  (probability {prob:.1%} ≥ threshold {THR:.1%})")
        else:
            st.success(f"🟢 **Low Risk Group**  (probability {prob:.1%} < threshold {THR:.1%})")
    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("View Input Parameters"):
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

st.sidebar.markdown("### ℹ️ Model Information")
st.sidebar.markdown(f"""
- Algorithm: **XGBoost**
- Number of Variables: **11**
- Decision Threshold: **{THR:.4f}**
- Task: Binary classification (Plasma Transfusion Yes/No)
""")
st.sidebar.warning(
    "⚠️ This tool is intended for **research and decision support only**. "
    "Final treatment decisions should be made by clinicians."
)
