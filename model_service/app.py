import os
import subprocess
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# --- Prometheus metrics ---
REQUESTS = Counter('predict_requests_total', 'Total number of predictions')
LATENCY = Histogram('predict_latency_seconds', 'Latency of prediction')

# --- Flask app ---
app = Flask(__name__)
swagger = Swagger(app)

# --- Paths ---
DATA_PATH = "data/raw/housing.csv"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# --- Generate synthetic data if missing ---
if not os.path.exists(DATA_PATH):
    X = np.random.randn(200, 8)
    y = X.dot(np.random.randn(8)) + np.random.randn(200) * 0.1
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df['target'] = y
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Generated synthetic data at {DATA_PATH}")

# --- Train model if missing ---
if not os.path.exists(MODEL_PATH):
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=['target'])
    y = data['target']
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Trained model and saved at {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded existing model from {MODEL_PATH}")

# --- Prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction using trained model
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: input
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
        description: Model predictions
        schema:
          type: array
          items:
            type: number
    """
    REQUESTS.inc()
    with LATENCY.time():
        data = pd.DataFrame(request.json)
        preds = model.predict(data)
        return jsonify(preds.tolist())

# --- Health check endpoint ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# --- Add Prometheus WSGI middleware ---
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    "/metrics": make_wsgi_app()
})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
