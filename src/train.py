import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import mlflow
from datetime import datetime

# -----------------------
# Paths
# -----------------------
PROCESSED_PATH = os.environ.get("PROCESSED_PATH", "data/processed/housing_processed.csv")
MLFLOW_ARTIFACTS_DIR = os.environ.get("MLFLOW_ARTIFACTS_DIR", "mlflow_artifacts")
os.makedirs(MLFLOW_ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MLFLOW_ARTIFACTS_DIR, "model.joblib")
METRICS_PATH = os.path.join(MLFLOW_ARTIFACTS_DIR, "metrics.json")

# -----------------------
# MLflow setup
# -----------------------
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("mlops-demo")

# -----------------------
# Load data
# -----------------------
if not os.path.exists(PROCESSED_PATH):
    raise FileNotFoundError(f"Processed data not found at {PROCESSED_PATH}")

data = pd.read_csv(PROCESSED_PATH)
X = data.drop("target", axis=1)
y = data["target"]

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Model training
# -----------------------
n_estimators = 50
max_depth = 10

with mlflow.start_run(run_name=f"run_{datetime.utcnow().isoformat()}"):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save model locally
    joblib.dump(model, MODEL_PATH)

    # Save metrics locally
    with open(METRICS_PATH, "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

    # Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")

    print(f"[INFO] Training done. MSE={mse:.4f}, R2={r2:.4f}")
