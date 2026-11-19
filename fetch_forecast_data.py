
import calendar
import time
from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

ACTUAL_DATA_FILE = DATA_DIR / "abuja_weather.csv"

OUTPUT_FILE = DATA_DIR / "abuja_forecast_training.csv"
EXCLUSIONS_FILE = DATA_DIR / "abuja_forecast_exclusions.csv"
SUMMARY_FILE = DATA_DIR / "abuja_forecast_coverage_summary.csv"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

LATITUDE = 9.0765
LONGITUDE = 7.3986
TIMEZONE = "Africa/Lagos"

FORECAST_VARIABLES = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
    "cloud_cover_previous_day1",
]

# Stop the script if the API cannot be reached after retries.
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
PAUSE_BETWEEN_REQUESTS = 1


# --------------------------------------------------
# 2. LOAD ACTUAL RAINFALL DATA
# --------------------------------------------------

print("=" * 60)
print("ABUJA HISTORICAL FORECAST DATA COLLECTION")
print("=" * 60)

if not ACTUAL_DATA_FILE.exists():
    raise FileNotFoundError(
        f"Actual weather dataset not found: {ACTUAL_DATA_FILE}"
    )

actual_df = pd.read_csv(ACTUAL_DATA_FILE)

required_actual_columns = [
    "time",
    "precipitation_sum",
    "rain",
]

missing_columns = [
    col for col in required_actual_columns
    if col not in actual_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in actual dataset: {missing_columns}"
    )

actual_df["time"] = pd.to_datetime(
    actual_df["time"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

actual_df["rain"] = pd.to_numeric(
    actual_df["rain"],
    errors="coerce"
)

actual_df["precipitation_sum"] = pd.to_numeric(
    actual_df["precipitation_sum"],
    errors="coerce"
)

actual_df = actual_df.dropna(
    subset=["time", "rain", "precipitation_sum"]
)

if actual_df["time"].duplicated().any():
    raise ValueError(
        "Duplicate dates found in the actual weather dataset."
    )

actual_df = actual_df[
    ["time", "precipitation_sum", "rain"]
].rename(
    columns={
        "precipitation_sum": "actual_precipitation_sum"
    }
)

actual_df["rain"] = actual_df["rain"].astype(int)

print("Actual rainfall dataset loaded.")
print("Available actual records:", len(actual_df))


# --------------------------------------------------
# 3. PREPARE MONTHLY DATE RANGES
# --------------------------------------------------

start = pd.Timestamp(START_DATE)
end = pd.Timestamp(END_DATE)

months = pd.period_range(
    start=start,
    end=end,
    freq="M"
)

all_daily_records = []
excluded_records = []
monthly_summary = []


# --------------------------------------------------
# 4. COLLECT HISTORICAL FORECASTS MONTH BY MONTH
# --------------------------------------------------

for month_number, month in enumerate(months, start=1):

    month_start = max(
        start,
        month.start_time
    ).strftime("%Y-%m-%d")

    month_end = min(
        end,
        month.end_time
    ).strftime("%Y-%m-%d")

    expected_dates = pd.date_range(
        start=month_start,
        end=month_end,
        freq="D"
    )

    print("\n" + "-" * 60)
    print(
        f"Month {month_number}/{len(months)}: "
        f"{month.strftime('%B %Y')}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(FORECAST_VARIABLES),
        "timezone": TIMEZONE,
        "start_date": month_start,
        "end_date": month_end,
    }

    response_data = None
    request_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()
            response_data = response.json()

            break

        except (
            requests.exceptions.RequestException,
            ValueError
        ) as error:

            request_error = str(error)

            print(
                f"Request attempt {attempt} failed: "
                f"{request_error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)

    # If the entire monthly request fails, record every date.
    if response_data is None:

        for date in expected_dates:
            excluded_records.append({
                "time": date.strftime("%Y-%m-%d"),
                "reason": "monthly_api_request_failed",
            })

        monthly_summary.append({
            "month": str(month),
            "expected_days": len(expected_dates),
            "valid_days": 0,
            "excluded_days": len(expected_dates),
            "status": "request_failed",
            "error": request_error,
        })

        print("Monthly request failed. Moving to next month.")

        time.sleep(PAUSE_BETWEEN_REQUESTS)
        continue

    hourly_data = response_data.get("hourly", {})

    if not hourly_data or "time" not in hourly_data:

        for date in expected_dates:
            excluded_records.append({
                "time": date.strftime("%Y-%m-%d"),
                "reason": "hourly_data_missing",
            })

        monthly_summary.append({
            "month": str(month),
            "expected_days": len(expected_dates),
            "valid_days": 0,
            "excluded_days": len(expected_dates),
            "status": "hourly_data_missing",
            "error": "",
        })

        print("No hourly data returned for this month.")

        time.sleep(PAUSE_BETWEEN_REQUESTS)
        continue

    # Check that every requested variable was returned.
    missing_variables = [
        variable for variable in FORECAST_VARIABLES
        if variable not in hourly_data
    ]

    if missing_variables:

        for date in expected_dates:
            excluded_records.append({
                "time": date.strftime("%Y-%m-%d"),
                "reason": (
                    "forecast_variables_missing: "
                    + ", ".join(missing_variables)
                ),
            })

        monthly_summary.append({
            "month": str(month),
            "expected_days": len(expected_dates),
            "valid_days": 0,
            "excluded_days": len(expected_dates),
            "status": "variables_missing",
            "error": ", ".join(missing_variables),
        })

        print("Required forecast variables are missing.")

        time.sleep(PAUSE_BETWEEN_REQUESTS)
        continue

    # Convert the hourly response into a DataFrame.
    hourly_df = pd.DataFrame(hourly_data)

    # Check that all returned columns have matching lengths.
    expected_length = len(hourly_df["time"])

    invalid_lengths = [
        variable for variable in FORECAST_VARIABLES
        if len(hourly_df[variable]) != expected_length
    ]

    if invalid_lengths:

        for date in expected_dates:
            excluded_records.append({
                "time": date.strftime("%Y-%m-%d"),
                "reason": "hourly_array_length_mismatch",
            })

        monthly_summary.append({
            "month": str(month),
            "expected_days": len(expected_dates),
            "valid_days": 0,
            "excluded_days": len(expected_dates),
            "status": "invalid_hourly_arrays",
            "error": ", ".join(invalid_lengths),
        })

        print("Hourly arrays have inconsistent lengths.")

        time.sleep(PAUSE_BETWEEN_REQUESTS)
        continue

    hourly_df["time"] = pd.to_datetime(
        hourly_df["time"],
        errors="coerce"
    )

    for variable in FORECAST_VARIABLES:
        hourly_df[variable] = pd.to_numeric(
            hourly_df[variable],
            errors="coerce"
        )

    hourly_df = hourly_df.dropna(subset=["time"])

    hourly_df["date"] = hourly_df["time"].dt.strftime(
        "%Y-%m-%d"
    )

    valid_days_this_month = 0
    excluded_days_this_month = 0

    # --------------------------------------------------
    # 5. VALIDATE AND AGGREGATE EACH DAY
    # --------------------------------------------------

    for date in expected_dates:

        date_string = date.strftime("%Y-%m-%d")

        day_df = hourly_df[
            hourly_df["date"] == date_string
        ].copy()

        # Require exactly 24 hourly records.
        if len(day_df) != 24:

            excluded_records.append({
                "time": date_string,
                "reason": (
                    f"expected_24_hours_received_{len(day_df)}"
                ),
            })

            excluded_days_this_month += 1
            continue

        # Require 24 distinct hourly timestamps.
        if day_df["time"].nunique() != 24:

            excluded_records.append({
                "time": date_string,
                "reason": "duplicate_or_missing_hourly_timestamps",
            })

            excluded_days_this_month += 1
            continue

        # Require all five variables to be complete.
        missing_counts = day_df[
            FORECAST_VARIABLES
        ].isna().sum()

        missing_variables_for_day = [
            variable
            for variable, count in missing_counts.items()
            if count > 0
        ]

        if missing_variables_for_day:

            excluded_records.append({
                "time": date_string,
                "reason": (
                    "missing_hourly_values: "
                    + ", ".join(missing_variables_for_day)
                ),
            })

            excluded_days_this_month += 1
            continue

        # Aggregate hourly forecast values into daily features.
        record = {
            "time": date_string,

            "temperature_max": day_df[
                "temperature_2m_previous_day1"
            ].max(),

            "temperature_min": day_df[
                "temperature_2m_previous_day1"
            ].min(),

            "temperature_mean": day_df[
                "temperature_2m_previous_day1"
            ].mean(),

            "humidity_mean": day_df[
                "relative_humidity_2m_previous_day1"
            ].mean(),

            "forecast_precipitation_sum": day_df[
                "precipitation_previous_day1"
            ].sum(),

            "wind_speed_max": day_df[
                "wind_speed_10m_previous_day1"
            ].max(),

            "cloud_cover_mean": day_df[
                "cloud_cover_previous_day1"
            ].mean(),
        }

        all_daily_records.append(record)
        valid_days_this_month += 1

    monthly_summary.append({
        "month": str(month),
        "expected_days": len(expected_dates),
        "valid_days": valid_days_this_month,
        "excluded_days": excluded_days_this_month,
        "status": "completed",
        "error": "",
    })

    print("Expected days:", len(expected_dates))
    print("Valid days:", valid_days_this_month)
    print("Excluded days:", excluded_days_this_month)

    time.sleep(PAUSE_BETWEEN_REQUESTS)


# --------------------------------------------------
# 6. MATCH FORECASTS WITH ACTUAL RAINFALL
# --------------------------------------------------

if not all_daily_records:
    raise RuntimeError(
        "No complete forecast records were collected. "
        "Check the API responses before continuing."
    )

forecast_df = pd.DataFrame(all_daily_records)

forecast_df = forecast_df.merge(
    actual_df,
    on="time",
    how="left",
    validate="one_to_one"
)

# Exclude dates without a matching actual rainfall label.
missing_actual = forecast_df["rain"].isna()

for date in forecast_df.loc[missing_actual, "time"]:
    excluded_records.append({
        "time": date,
        "reason": "actual_rainfall_label_missing",
    })

forecast_df = forecast_df.loc[
    ~missing_actual
].copy()

forecast_df["rain"] = forecast_df["rain"].astype(int)

# Useful diagnostic for comparison with the actual rainfall label.
# This is a forecast-based threshold, not the ML model prediction.
forecast_df["forecast_rain"] = (
    forecast_df["forecast_precipitation_sum"] > 0.1
).astype(int)

forecast_df = forecast_df.sort_values(
    "time"
).reset_index(drop=True)


# --------------------------------------------------
# 7. SAVE DATASETS AND COVERAGE REPORTS
# --------------------------------------------------

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)

exclusions_df = pd.DataFrame(
    excluded_records,
    columns=["time", "reason"]
)

exclusions_df.to_csv(
    EXCLUSIONS_FILE,
    index=False
)

summary_df = pd.DataFrame(monthly_summary)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# --------------------------------------------------
# 8. FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLLECTION COMPLETED")
print("=" * 60)

print("\nForecast records saved:", len(forecast_df))
print("Excluded dates:", len(exclusions_df))

print("\nActual rainfall class distribution:")
print(forecast_df["rain"].value_counts().sort_index())

print("\nForecast rainfall class distribution:")
print(forecast_df["forecast_rain"].value_counts().sort_index())

print("\nFinal date range:")

if not forecast_df.empty:
    print("First date:", forecast_df["time"].min())
    print("Last date:", forecast_df["time"].max())

print("\nSaved files:")
print(OUTPUT_FILE)
print(EXCLUSIONS_FILE)
print(SUMMARY_FILE)

print("\nMonthly coverage:")
print(summary_df.to_string(index=False))

print("\nForecast dataset preview:")
print(forecast_df.head())

print("\nForecast dataset information:")
print(forecast_df.info())

print("\nCollection finished.")
