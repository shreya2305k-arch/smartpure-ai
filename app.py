
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartPure AI", page_icon="💧", layout="wide")

q_model = joblib.load("water_quality_model.joblib")
f_model = joblib.load("filter_life_model.joblib")

st.title("💧 SmartPure AI")
st.subheader("ML-Based Water Quality & Filter Life Prediction System")
st.caption("Internship prototype using a synthetic dataset for demonstration. Replace with approved company/sensor data for real deployment.")

st.sidebar.header("Water & Usage Inputs")
ph = st.sidebar.slider("pH", 4.5, 9.5, 7.1, 0.1)
tds = st.sidebar.slider("TDS (ppm)", 80, 1100, 420, 10)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.1, 12.0, 2.0, 0.1)
hardness = st.sidebar.slider("Hardness (mg/L)", 40, 450, 180, 5)
conductivity = st.sidebar.slider("Conductivity (µS/cm)", 100, 1800, 610, 10)
temperature = st.sidebar.slider("Temperature (°C)", 15.0, 38.0, 26.0, 0.5)
daily_usage = st.sidebar.slider("Daily usage (L)", 2.0, 30.0, 12.0, 0.5)
users = st.sidebar.slider("Number of users", 1, 8, 4)
filter_age = st.sidebar.slider("Filter age (days)", 10, 360, 120, 5)

quality_input = pd.DataFrame([{
    "pH": ph, "TDS_ppm": tds, "Turbidity_NTU": turbidity,
    "Hardness_mgL": hardness, "Conductivity_uScm": conductivity,
    "Temperature_C": temperature
}])
filter_input = pd.DataFrame([{
    "TDS_ppm": tds, "Daily_Usage_L": daily_usage,
    "Users": users, "Filter_Age_Days": filter_age
}])

quality_pred = q_model.predict(quality_input)[0]
quality_prob = q_model.predict_proba(quality_input).max() * 100
filter_days = max(1, round(float(f_model.predict(filter_input)[0])))

if filter_days < 30:
    status = "Replacement recommended soon"
elif filter_days < 60:
    status = "Monitor filter condition"
else:
    status = "Filter condition appears normal"

c1, c2, c3 = st.columns(3)
c1.metric("Water Quality", quality_pred)
c2.metric("Prediction Confidence", f"{quality_prob:.1f}%")
c3.metric("Estimated Filter Life", f"{filter_days} days")

st.divider()
st.subheader("Maintenance Recommendation")
if filter_days < 30:
    st.error("🔴 " + status)
elif filter_days < 60:
    st.warning("🟡 " + status)
else:
    st.success("🟢 " + status)

st.subheader("Input Summary")
st.dataframe(pd.DataFrame({
    "Parameter": ["pH","TDS","Turbidity","Hardness","Daily Usage","Users","Filter Age"],
    "Value": [ph,f"{tds} ppm",f"{turbidity} NTU",f"{hardness} mg/L",f"{daily_usage} L",users,f"{filter_age} days"]
}), use_container_width=True)

st.info("This is a proof-of-concept. Water safety decisions should not be made from this prototype alone; validated laboratory measurements and company-approved engineering thresholds are required.")
