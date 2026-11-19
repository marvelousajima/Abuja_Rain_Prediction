
import requests
import pandas as pd

# Abuja coordinates
LATITUDE = 9.0765
LONGITUDE = 7.3986

URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "shortwave_radiation_sum"
    ],
    "timezone": "Africa/Lagos"
}

print("Connecting to Open-Meteo API...")

try:
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()

    weather_data = response.json()

    df = pd.DataFrame(weather_data["daily"])

    # Rain = 1, No rain = 0
    df["rain"] = (df["precipitation_sum"] > 0.1).astype(int)

    df.to_csv("data/abuja_weather.csv", index=False)

    print("\nData successfully retrieved!")
    print("Dataset shape:", df.shape)

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nRainfall distribution:")
    print(df["rain"].value_counts())

    print("\nDataset saved successfully!")

except requests.exceptions.RequestException as error:
    print("API connection error:", error)

except (KeyError, ValueError) as error:
    print("Data processing error:", error)
