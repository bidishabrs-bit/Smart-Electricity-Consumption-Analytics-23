# Smart Electricity Consumption Analytics

A Python + Streamlit + Machine Learning project for exploring household electricity use and estimating daily consumption.

## Features
- Overview dashboard with KPIs and monthly/daily consumption charts
- Interactive daily electricity prediction in kWh
- Indicative daily and 30-day electricity cost calculation
- Dataset explorer with downloadable CSV and correlation view
- Model performance metrics and feature importance
- General energy-saving suggestions

## Technology
Python, Pandas, NumPy, Scikit-learn, Random Forest Regression, Streamlit, Plotly, Joblib.

## Dataset
The repository includes **3,650 synthetic daily records**. Inputs include temperature, humidity, occupants, AC hours, washing cycles, lighting hours, device hours, previous-day consumption, day of week, month, fan hours and refrigerator hours. The target is `daily_kwh`.

**Important:** This is synthetic demonstration data, not actual smart-meter data. Model scores are not evidence of real-world performance.

## Model
- Algorithm: `RandomForestRegressor`
- Trees: 120
- Maximum depth: 14
- Train/test split: 80:20
- Random state: 42
- Evaluation: MAE, RMSE and R²

### Current test results
- MAE: 0.7841 kWh
- RMSE: 0.9847 kWh
- R²: 0.8247

These results are calculated from the included synthetic dataset and may vary if the dataset or training settings change.

## Run locally (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Repository structure
```text
Smart_Electricity_Consumption_Analytics/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── energy_consumption.csv
├── models/
│   └── energy_model.joblib
└── reports/
    └── metrics.json
```

## Limitations and future work
- Replace synthetic records with verified smart-meter data.
- Validate the model across different households and regions.
- Add time-series forecasting and anomaly alerts.
- Account for tariff slabs, fixed charges and taxes in bill estimation.
- Integrate IoT meters for real-time readings.

## Academic integrity
This is an independently assembled educational prototype. Cite any external datasets, libraries, tutorials or repositories that you use. Do not describe synthetic data as real household measurements.
