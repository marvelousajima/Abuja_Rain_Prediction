# Abuja Rain Prediction

A machine-learning project that estimates whether rain is expected in Abuja, Nigeria. The project combines a Random Forest classifier trained on historical one-day-ahead forecast data with a separate historical monthly rainfall estimate for dates outside the live forecast window.

## Project overview

The application has two prediction approaches:

1. **Next-day machine-learning prediction** — uses forecast meteorological variables as inputs to a trained Random Forest classifier. The model's rain classification threshold is **0.65**.
2. **Historical seasonal estimate** — for dates outside the supported next-day prediction path, the application calculates the historical frequency of rainy days for the selected calendar month using Abuja observations from 2015–2025. This is a seasonal estimate, not a date-specific weather forecast.

The project defines a rainy day as a day with more than **0.1 mm** of precipitation.

## Model performance

On the held-out chronological test set (143 days), the Random Forest model using a 0.65 probability threshold achieved:

- Accuracy: **92.31%**
- Balanced accuracy: **92.41%**
- Confusion matrix: **68 true negatives, 2 false positives, 9 false negatives, 64 true positives**

The original forecast-based rain/no-rain baseline achieved **94.41% accuracy** on the same test set. Therefore, the project does not claim that the machine-learning model outperformed the original forecast baseline.

## Repository structure

The repository is expected to contain the following files and folders:

```text
abuja-rain-prediction/
├── app.py
├── fetch_data.py
├── fetch_forecast_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── abuja_weather.csv
│   ├── abuja_forecast_training.csv
│   ├── abuja_forecast_exclusions.csv
│   └── abuja_forecast_coverage_summary.csv
├── models/
│   ├── random_forest_model.pkl
│   └── model_info.pkl
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_development.ipynb
```

Keep the `data/` and `models/` folders in the repository if the deployed application reads the CSV and model files from those locations. In particular, the Streamlit app requires:

- `models/random_forest_model.pkl`
- `models/model_info.pkl`
- `data/abuja_weather.csv` (for the historical seasonal estimate)

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/abuja-rain-prediction.git
cd abuja-rain-prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Push the complete project repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and create a new app.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy the app.

Ensure the `requirements.txt`, `models/` files, and required `data/` file are committed to the selected GitHub branch. Do not commit passwords, API keys, private credentials, or `.streamlit/secrets.toml`.

## Data sources

- Historical daily weather observations: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Historical forecast runs: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- Live forecast data: [Open-Meteo Forecast API](https://open-meteo.com/en/docs)

Location: Abuja, Nigeria (latitude 9.0765, longitude 7.3986). Time zone: Africa/Lagos.

## Important methodological note

The model was trained using historical forecast-run variables corresponding to the previous-day forecast lead time. Before presenting a public deployment as fully validated, confirm that the live forecast request in `app.py` supplies features at the same forecast lead time and with the same aggregation definitions as the training data. A live forecast for a selected date is not automatically equivalent to a forecast issued exactly 24 hours before that date.

The historical monthly estimate is provided for seasonal context only. It should not be interpreted as a precise daily weather prediction.

## Reproducibility

The saved model and metadata are stored in `models/`. The metadata file records the model feature order and classification threshold. If the model is retrained, update both saved model files together and verify that the app uses the same feature definitions.
