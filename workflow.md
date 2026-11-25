# ML workflow

1. Load `data/energy_consumption.csv` with Pandas.
2. Inspect data types, missing values and summary statistics.
3. Separate the eight predictor features from `daily_kwh`.
4. Split into training and test sets (80/20).
5. Fit a Random Forest Regressor (120 trees).
6. Evaluate with MAE, RMSE and R².
7. Save the trained model with Joblib; load it in the Streamlit app.

Run the reproducible training script from the repository root:

```bash
python train_model.py
```
