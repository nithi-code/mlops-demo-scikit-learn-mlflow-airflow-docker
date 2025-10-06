import json
import mlflow
from datetime import datetime
import os

METRICS_PATH = "artifacts/metrics.json"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("mlops-demo")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

print("Evaluation metrics:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"R2: {metrics['r2']:.4f}")

# Optionally log metrics to MLflow as a new run
with mlflow.start_run(run_name=f"eval_{datetime.utcnow().isoformat()}"):
    mlflow.log_metrics(metrics)
