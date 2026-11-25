from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "energy_consumption.csv")
features = [c for c in df.columns if c not in ["date", "daily_kwh"]]
X, y = df[features], df["daily_kwh"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "mae_kwh": round(float(mean_absolute_error(y_test, pred)), 4),
    "rmse_kwh": round(float(mean_squared_error(y_test, pred) ** 0.5), 4),
    "r2": round(float(r2_score(y_test, pred)), 4),
    "records": int(len(df)),
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "features": features,
    "note": "Synthetic demonstration data only; not validated for real-world use."
}
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": features}, ROOT / "models" / "energy_model.joblib")
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Model trained and saved.")
print(json.dumps(metrics, indent=2))
