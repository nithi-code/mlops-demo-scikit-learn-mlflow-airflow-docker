
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os, json
from datetime import datetime

os.makedirs("artifacts", exist_ok=True)

data = pd.read_csv("data/raw/housing.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("mlops-demo")

with mlflow.start_run(run_name=f"run_{datetime.utcnow().isoformat()}") as run:
    n_estimators = 50
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    model_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    # write metrics file
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)
    print("Training finished. MSE:", mse, "R2:", r2)
