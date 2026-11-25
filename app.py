from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "energy_consumption.csv"
MODEL_PATH = ROOT / "models" / "energy_model.joblib"

st.set_page_config(page_title="Smart Electricity Consumption Analytics", page_icon="⚡", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_model()
model, features = bundle["model"], bundle["features"]

st.sidebar.title("⚡ Energy Analytics")
page = st.sidebar.radio("Navigate", [
    "Overview", "Predict Consumption", "Explore Dataset",
    "Model Performance", "Energy Saving Tips"
])
st.sidebar.caption("Educational prototype • synthetic demonstration data")

if page == "Overview":
    st.title("Smart Electricity Consumption Analytics")
    st.write("Explore household electricity patterns and estimate daily energy demand.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Average daily use", f"{df.daily_kwh.mean():.2f} kWh")
    c3.metric("Peak daily use", f"{df.daily_kwh.max():.2f} kWh")
    c4.metric("Model", "Random Forest")
    monthly = df.assign(month=df.date.dt.to_period("M").astype(str)).groupby("month", as_index=False).daily_kwh.mean()
    st.subheader("Average daily consumption by month")
    st.plotly_chart(px.line(monthly, x="month", y="daily_kwh", markers=True,
                            labels={"month":"Month","daily_kwh":"Average daily use (kWh)"}),
                    use_container_width=True)
    st.subheader("Consumption distribution")
    st.plotly_chart(px.histogram(df, x="daily_kwh", nbins=35,
                                 labels={"daily_kwh":"Daily consumption (kWh)"}),
                    use_container_width=True)

elif page == "Predict Consumption":
    st.title("Predict Daily Electricity Consumption")
    st.write("Enter household and appliance-use details to estimate daily electricity demand.")
    a,b,c = st.columns(3)
    values = {}
    with a:
        values["temperature_c"] = st.slider("Temperature (°C)", 12.0, 45.0, 30.0)
        values["humidity_pct"] = st.slider("Humidity (%)", 20.0, 95.0, 60.0)
        values["occupants"] = st.number_input("Number of occupants", 1, 10, 3)
        values["ac_hours"] = st.slider("AC usage (hours/day)", 0.0, 18.0, 4.0)
    with b:
        values["washing_cycles"] = st.number_input("Washing cycles/day", 0, 5, 1)
        values["lighting_hours"] = st.slider("Lighting (hours/day)", 1.0, 14.0, 6.0)
        values["device_hours"] = st.slider("Electronic device use (hours/day)", 1.0, 18.0, 7.0)
        values["previous_day_kwh"] = st.number_input("Previous-day use (kWh)", 3.0, 22.0, 9.5)
    with c:
        values["day_of_week"] = st.selectbox("Day of week (Monday=0)", list(range(7)), index=2)
        values["month"] = st.selectbox("Month", list(range(1,13)), index=5)
        values["fan_hours"] = st.slider("Fan usage (hours/day)", 1.0, 20.0, 8.0)
        values["refrigerator_hours"] = st.slider("Refrigerator operation (hours/day)", 22.0, 24.0, 24.0)
    tariff = st.number_input("Electricity tariff (₹ per kWh)", min_value=0.0, value=8.0, step=0.5)
    if st.button("Predict Daily Consumption", type="primary"):
        row = pd.DataFrame([[values[f] for f in features]], columns=features)
        estimate = max(0.0, float(model.predict(row)[0]))
        daily_cost = estimate * tariff
        st.success(f"Estimated daily consumption: {estimate:.2f} kWh")
        x,y = st.columns(2)
        x.metric("Indicative daily cost", f"₹{daily_cost:,.2f}")
        y.metric("Indicative 30-day cost", f"₹{daily_cost*30:,.2f}")
        st.caption("Cost is an estimate based only on the entered per-unit tariff. Fixed charges, taxes and slab rates are excluded.")

elif page == "Explore Dataset":
    st.title("Explore the Dataset")
    st.write("The included dataset contains 3,650 synthetic daily household records.")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Feature relationships")
    corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix"),
                    use_container_width=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False), "energy_consumption.csv", "text/csv")

elif page == "Model Performance":
    st.title("Model Performance")
    metrics_path = ROOT / "reports" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE", f"{metrics.get('mae_kwh', '—')} kWh")
    m2.metric("RMSE", f"{metrics.get('rmse_kwh', '—')} kWh")
    m3.metric("R²", metrics.get("r2", "—"))
    st.write(f"Training records: {metrics.get('train_records', '—')} | Test records: {metrics.get('test_records', '—')}")
    importance = pd.DataFrame({"Feature":features, "Importance":model.feature_importances_}).sort_values("Importance", ascending=True)
    st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h",
                           title="Random Forest feature importance"), use_container_width=True)
    st.warning("These results are calculated using synthetic demonstration data. They do not establish real-world predictive accuracy.")

else:
    st.title("Energy Saving Tips")
    st.write("General suggestions based on common household energy practices:")
    st.markdown("""
    - Set air-conditioning to a moderate temperature and keep doors and windows closed.
    - Switch off lights, fans and devices when they are not needed.
    - Prefer LED lighting and energy-efficient appliances.
    - Run washing machines with full loads where practical.
    - Clean AC filters regularly to maintain efficient operation.
    - Track daily usage and compare it with your household's normal pattern.
    """)
    st.info("Recommendations are general guidance, not personalised energy audits.")

st.divider()
st.caption("Smart Electricity Consumption Analytics | Student project prototype | Synthetic data")
