"""SmartPure AI - Streamlit Dashboard"""
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartPure AI", page_icon="💧", layout="centered")

# Train automatically the first time (useful for online deployment)
if not (os.path.exists("quality_model.pkl") and os.path.exists("filter_model.pkl")):
    import train_models  # noqa: F401  (running the file trains and saves models)

quality_model = joblib.load("quality_model.pkl")
filter_model = joblib.load("filter_model.pkl")

st.title("💧 SmartPure AI")
st.caption("ML-Based Water Quality & Filter Life Prediction System")

st.sidebar.header("Enter Water & Usage Details")
ph = st.sidebar.slider("pH", 4.0, 10.0, 7.2, 0.1)
tds = st.sidebar.slider("TDS (mg/L)", 20, 1500, 250)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 15.0, 1.5, 0.1)
daily_usage = st.sidebar.number_input("Daily usage (litres)", 10, 500, 100)
users = st.sidebar.number_input("Number of users", 1, 15, 4)
filter_age = st.sidebar.number_input("Filter age (days)", 0, 600, 90)

if st.sidebar.button("Predict", type="primary"):
    # ----- Water quality -----
    q_input = pd.DataFrame([[ph, tds, turbidity]], columns=["ph", "tds", "turbidity"])
    q = int(quality_model.predict(q_input)[0])
    confidence = quality_model.predict_proba(q_input)[0].max() * 100
    labels = {0: "✅ Safe to drink", 1: "⚠️ Moderate - needs attention", 2: "❌ Unsafe - do not drink"}

    st.subheader("1. Water Quality")
    if q == 0:
        st.success(labels[q])
    elif q == 1:
        st.warning(labels[q])
    else:
        st.error(labels[q])
    st.write(f"Model confidence: **{confidence:.0f}%**")

    # ----- Filter life -----
    f_input = pd.DataFrame(
        [[tds, turbidity, daily_usage, users, filter_age]],
        columns=["tds", "turbidity", "daily_usage", "users", "filter_age"],
    )
    days = max(0, int(filter_model.predict(f_input)[0]))

    st.subheader("2. Filter Life")
    col1, col2 = st.columns(2)
    col1.metric("Remaining life", f"{days} days")
    col2.metric("Approx. months", f"{days/30:.1f}")

    if days <= 15:
        st.error("🔴 Replace the filter very soon!")
    elif days <= 45:
        st.warning("🟡 Plan a filter replacement soon.")
    else:
        st.success("🟢 Filter is in good condition.")
    st.progress(min(days, 365) / 365)

    # ----- Why? (feature importance) -----
    with st.expander("Which inputs matter most for filter life?"):
        imp = pd.Series(filter_model.feature_importances_, index=f_input.columns)
        st.bar_chart(imp.sort_values())
else:
    st.info("👈 Enter the values in the sidebar and click **Predict**.")
