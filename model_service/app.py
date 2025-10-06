import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from datetime import datetime

# -------------------------
# Flask + Swagger setup
# -------------------------
app = Flask(__name__)
swagger = Swagger(app)

# -------------------------
# Prometheus metrics
# -------------------------
REQUESTS = Counter('predict_requests_total', 'Total number of predictions')
LATENCY = Histogram('predict_latency_seconds', 'Latency of prediction')

# -------------------------
# Paths
# -------------------------
DATA_PATH = "data/raw/housing.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# -------------------------
# MLflow setup
# -------------------------
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("mlops-demo")

# -------------------------
# Load or train model
# -------------------------
if not os.path.exists(MODEL_PATH):
    print("[INFO] Training new model...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Make sure dataset exists.")

    data = pd.read_csv(DATA_PATH)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    n_estimators = 50
    max_depth = 10
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[INFO] Model trained. MSE={mse:.4f}, R2={r2:.4f}")

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save metrics locally
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"mse": mse, "r2": r2}, f)

    # Log to MLflow
    with mlflow.start_run(run_name=f"run_{datetime.utcnow().isoformat()}"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

else:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded existing model from {MODEL_PATH}")

# -------------------------
# Prediction endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Make prediction using trained RandomForest model
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: input
        description: JSON array of input feature dicts
        required: true
        schema:
          type: array
          items:
            type: object
            properties:
              feature_0: {type: number}
              feature_1: {type: number}
              feature_2: {type: number}
              feature_3: {type: number}
              feature_4: {type: number}
              feature_5: {type: number}
              feature_6: {type: number}
              feature_7: {type: number}
    responses:
      200:
        description: Prediction results
    """
    REQUESTS.inc()
    import time
    start = time.time()

    try:
        expected_features = list(model.feature_names_in_)
        data = pd.DataFrame(request.json)
        data = data[expected_features]  # Align input columns
    except KeyError as e:
        return jsonify({
            "error": "Feature mismatch",
            "message": str(e),
            "expected_features": expected_features
        }), 400

    preds = model.predict(data)
    LATENCY.observe(time.time() - start)
    return jsonify(predictions=preds.tolist())

# -------------------------
# Health endpoint
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -------------------------
# Prometheus metrics endpoint
# -------------------------
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    "/metrics": make_wsgi_app()
})

# -------------------------
# Root endpoint
# -------------------------
@app.route("/")
def index():
    return jsonify({
        "message": "MLOps demo model service running",
        "swagger": "/apidocs",
        "predict_endpoint": "/predict"
    })

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
