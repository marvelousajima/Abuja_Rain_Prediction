
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date, timedelta

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Abuja Rain Prediction",
    page_icon="🌧️",
    layout="wide"
)

# --------------------------------
# CUSTOM APPLICATION STYLING
# --------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #172B4D;
        font-weight: 750;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #243B53;
    }

    .intro-text {
        color: #64748B;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        min-height: 45px;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
        border: none;
    }

    div[data-testid="stDateInput"] input {
        border-radius: 9px;
    }

    .result-label {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD SAVED MODEL
# --------------------------------

MODEL_PATH = "models/random_forest_model.pkl"
INFO_PATH = "models/model_info.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model_info = joblib.load(INFO_PATH)
    return model, model_info

model, model_info = load_model()

features = model_info["features"]
threshold = model_info["threshold"]


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("Abuja Rain Prediction")

st.markdown(
    """
    <p class="intro-text">
        Explore rainfall predictions for Abuja, Nigeria,
        using machine learning and historical weather patterns.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# DATE SELECTION
# --------------------------------

st.subheader("Choose a prediction date")

today = date.today()
last_available_date = today + timedelta(days=365)

selected_date = st.date_input(
    "Select a date",
    value=today,
    min_value=today,
    max_value=last_available_date,
    format="DD/MM/YYYY"
)

st.caption(
    f"Selected date: {selected_date.strftime('%A, %d %B %Y')}"
)

st.divider()




# --------------------------------
# DATE SELECTION
# --------------------------------

st.subheader("Select Prediction Date")

today = date.today()
last_available_date = today + timedelta(days=365)

selected_date = st.date_input(
    "Choose a date for your rainfall prediction",
    value=today,
    min_value=today,
    max_value=last_available_date,
    format="DD/MM/YYYY"
)

st.write(
    "Selected date:",
    selected_date.strftime("%A, %d %B %Y")
)



# --------------------------------
# LIVE WEATHER FORECAST
# --------------------------------

import requests

def get_live_forecast_features(selected_date):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 9.0765,
        "longitude": 7.3986,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover"
        ],
        "forecast_days": 16,
        "timezone": "Africa/Lagos"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    weather_data = response.json()

    hourly_data = pd.DataFrame(weather_data["hourly"])

    # Keep only records for the selected date
    hourly_data = hourly_data[
        hourly_data["time"].str.startswith(selected_date.isoformat())
    ]

    # We need all 24 hourly records
    if len(hourly_data) != 24:
        return None

    # Check for missing weather values
    required_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "cloud_cover"
    ]

    if hourly_data[required_columns].isnull().any().any():
        return None

    # Calculate the same daily features used during model training
    daily_features = {
        "temperature_max": hourly_data["temperature_2m"].max(),
        "temperature_min": hourly_data["temperature_2m"].min(),
        "temperature_mean": hourly_data["temperature_2m"].mean(),
        "humidity_mean": hourly_data["relative_humidity_2m"].mean(),
        "forecast_precipitation_sum": hourly_data["precipitation"].sum(),
        "wind_speed_max": hourly_data["wind_speed_10m"].max(),
        "cloud_cover_mean": hourly_data["cloud_cover"].mean()
    }

    # Arrange features in the model's expected order
    features_df = pd.DataFrame([daily_features])
    features_df = features_df[features]

    return features_df

# --------------------------------
# HISTORICAL SEASONAL ESTIMATE
# --------------------------------

@st.cache_data
def load_historical_data():

    historical_data = pd.read_csv("data/abuja_weather.csv")

    historical_data["time"] = pd.to_datetime(
        historical_data["time"]
    )

    historical_data["month"] = (
        historical_data["time"].dt.month
    )

    return historical_data


def get_seasonal_estimate(selected_date):

    historical_data = load_historical_data()

    selected_month = selected_date.month

    monthly_data = historical_data[
        historical_data["month"] == selected_month
    ]

    rain_probability = monthly_data["rain"].mean()

    return rain_probability




# --------------------------------
# GENERATE RAINFALL PREDICTION
# --------------------------------

st.subheader("Rainfall Prediction")

if st.button("Generate Prediction", type="primary", use_container_width=True):

    tomorrow = date.today() + timedelta(days=1)

    if selected_date == tomorrow:

        try:
            with st.spinner("Analysing forecast weather data..."):

                forecast_features = get_live_forecast_features(
                    selected_date
                )

            if forecast_features is None:

                st.warning(
                    "Complete hourly forecast data is unavailable "
                    "for this date. Please try again later."
                )

            else:

                rain_score = model.predict_proba(
                    forecast_features
                )[0][1]

                prediction = int(rain_score >= threshold)

                st.markdown("### Prediction Result")

                if prediction == 1:
                    st.success("🌧️ Rain Expected")
                else:
                    st.info("☀️ No Rain Expected")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Model Rain Score",
                        f"{rain_score * 100:.1f}%"
                    )

                    st.progress(float(rain_score))

                with col2:
                    st.metric(
                        "Classification Threshold",
                        f"{threshold * 100:.0f}%"
                    )

                    st.progress(float(threshold))

                st.caption(
                    "Prediction Approach: Machine learning-based next-day rainfall "
                    "prediction for Abuja using forecast meteorological variables "
                    "processed through a trained Random Forest classification model. "
                    "The reported rain score represents the model's estimated "
                    "likelihood of rainfall based on the supplied forecast conditions."
                )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Unable to retrieve weather data: {error}"
            )

        except Exception as error:
            st.error(
                f"An unexpected error occurred: {error}"
            )

    else:

        try:
            seasonal_probability = get_seasonal_estimate(
                selected_date
            )

            st.markdown("### Prediction Result")

            if seasonal_probability >= 0.50:
                st.success("🌧️ Rain Expected")
            else:
                st.info("☀️ No Rain Expected")

            st.metric(
                "Historical Seasonal Rain Frequency",
                f"{seasonal_probability * 100:.1f}%"
            )

            st.progress(float(seasonal_probability))

            st.caption(
                "Prediction Approach: Historical rainfall probability estimation "
                "for Abuja using observed monthly rainfall patterns from 2015–2025. "
                "The estimate reflects the historical frequency of rainfall "
                "during the selected month and provides seasonal context rather "
                "than a date-specific weather forecast."
            )

        except Exception as error:
            st.error(
                f"Unable to generate seasonal estimate: {error}"
            )


