# MLOps Demo: Scikit-learn + Docker + MLflow + Airflow + Prometheus + Grafana

![Python](https://img.shields.io/badge/python-3.10-blue)
![Docker](https://img.shields.io/badge/docker-20.10.17-blue)
![MLflow](https://img.shields.io/badge/mlflow-2.6.3-orange)
![Airflow](https://img.shields.io/badge/airflow-2.6.3-red)

This project demonstrates a full **MLOps pipeline** using a machine learning model trained on the Boston Housing dataset, deployed via Docker, monitored with Prometheus/Grafana, tracked using MLflow, and orchestrated with Airflow.

---

## Project Structure

```
mlops-demo-scikit-learn-mlflow-airflow-docker/
├── data/
│   └── raw/
│       └── housing.csv          # Boston Housing dataset or synthetic dataset
├── artifacts/                   # Model and metrics artifacts
├── mlflow_server/
│   └── Dockerfile               # Dockerfile for MLflow server
├── model_service/
│   ├── Dockerfile               # Dockerfile for model service
│   └── app.py                   # Flask + Swagger + Prometheus model API
├── airflow_dags/                # Airflow DAGs
├── prometheus/
│   └── prometheus.yml           # Prometheus configuration
├── grafana/
│   └── dashboard.json           # Grafana dashboard config
├── src/
│   └── train.py                 # Model training script
├── docker-compose.yml
└── README.md
```

---

## Dataset

The project uses the **Boston Housing dataset** (or a synthetic version generated in `data/raw/housing.csv`) for predicting housing prices. Each row corresponds to one data point (house/neighborhood) and contains the following features:

| Feature           | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| feature_0 (CRIM)  | Per capita crime rate by town                                            |
| feature_1 (ZN)    | Proportion of residential land zoned for lots over 25,000 sq.ft          |
| feature_2 (INDUS) | Proportion of non-retail business acres per town (industrial area)       |
| feature_3 (CHAS)  | Charles River dummy variable (1 if tract bounds river, 0 otherwise)      |
| feature_4 (NOX)   | Nitric oxides concentration (parts per 10 million) – pollution indicator |
| feature_5 (RM)    | Average number of rooms per dwelling                                     |
| feature_6 (AGE)   | Proportion of owner-occupied units built prior to 1940                   |
| feature_7 (DIS)   | Weighted distances to five Boston employment centers                     |
| target            | House price (continuous value)                                           |

**Example row:**

```json
[
  {
    "feature_0": 0.4967141530112327,
    "feature_1": -0.13826430117118466,
    "feature_2": 0.6476885381006925,
    "feature_3": 1.5230298564080254,
    "feature_4": -0.23415337472333597,
    "feature_5": -0.23413695694918055,
    "feature_6": 1.5792128155073915,
    "feature_7": 0.7674347291529088,
    "target": -0.10740984079186282
  }
]
```

---

## Services

| Service    | URL                     |
| ---------- | ----------------------- |
| Model API  | `http://localhost:8000` |
| MLflow     | `http://localhost:5000` |
| Airflow    | `http://localhost:8080` |
| Prometheus | `http://localhost:9090` |
| Grafana    | `http://localhost:3000` |
| Jenkins    | `http://localhost:8081/jenkins` |

* **Model Service:** Flask app serving `/predict` endpoint with Swagger UI and Prometheus metrics.
* **MLflow:** Model tracking server.
* **Airflow:** Orchestration for pipelines.
* **Prometheus:** Metrics collection.
* **Grafana:** Visualization dashboards.
* **Jenkins:** CI/CD server.

---

## API Endpoints

### Predict

* **URL:** `/predict`
* **Method:** `POST`
* **Body:** JSON array of input feature objects:

```json
[
  {
    "feature_0": 0.4967,
    "feature_1": -0.1383,
    "feature_2": 0.6477,
    "feature_3": 1.5230,
    "feature_4": -0.2342,
    "feature_5": -0.2341,
    "feature_6": 1.5792,
    "feature_7": 0.7674
  }
]
```

* **Response:**

```json
{
  "predictions": [
    -0.1049
  ]
}
```

### Health Check

* **URL:** `/health`
* **Method:** `GET`
* **Response:**

```json
{
  "status": "ok"
}
```

---

## Running the Project

1. Build and start all services:

    ```bash
    docker-compose up -d --build
    ```

2. Access services as listed above.
3. Swagger UI for model API: `http://localhost:8000/apidocs`

## Useful Docker Commands

* Train Model (Generates artifacts/model.joblib) - **docker-compose run --rm model-service python src/train.py**
* Restart Model-Service - **docker-compose up -d --build model-service**
* Log Check - **docker-compose logs -f model-service**
* Jenkins Password generator - **docker exec -it mlops-demo-scikit-learn-mlflow-airflow-docker-jenkins-1 cat /var/jenkins_home/secrets/initialAdminPassword**

---

## DVC Explanation

1. Prepare_Data stage

    * Reads raw data (data/raw/housing.csv) and preprocesses it (e.g., filling missing values, scaling, encoding).
    * Outputs processed data in data/processed/housing_processed.csv.

2. Train Stage

    * Takes processed data, trains the RandomForestRegressor, saves model.joblib, logs metrics/artifacts to MLflow and outputs:

        **artifacts/model.joblib → trained model**
        **artifacts/metrics.json → training metrics (MSE, R2)**

3. Evaluate Stage

    * Uses the trained model to evaluate performance on test/validation data.
    * Outputs evaluation results in artifacts/evaluation.json.
    * Reads metrics and optionally logs evaluation run to MLflow

4. Key Points

    * Every train.py run creates a new MLflow run with timestamped run names.
    * Parameters, metrics, and the model artifact are logged.
    * evaluate.py can optionally log metrics to MLflow as a separate evaluation run.
    * Make sure your MLflow server (MLFLOW_TRACKING_URI) is running and accessible from the container.
    * Every run is reproducible. dvc repro executes stages if inputs change
    * MLflow runs are automatically linked because train.py logs artifacts and metrics.

    ## Steps to integrate DVC:

    ```bash
    # Initialize DVC in your project
    dvc init

    # Add raw dataset to DVC
    dvc add data/raw/housing.csv

    # Run the full pipeline
    dvc repro
    ```
 * This will track all dependencies, outputs, and allow you to reproduce the entire pipeline anytime.

---

## Jenkins automates the full workflow

  After code push, Jenkins reproduces DVC pipeline → trains model → deploys services → validates API → logs results.

  ## Stages:
  1. **Checkout code** - 
  2. **Setup Python Environment**
  3. **Prepare Data / DVC** - skips if no .dvc or remote configured. Prevents failures in Jenkins.
  4. **Wait for MLflowt** - ensures train.py can connect
  5. **Train model (via your train script)**
  6. **Build & Deploy Docker Services**
  7. **Test Model Prediction**  - Sends a sample JSON to your /predict endpoint.Captures the response from the model API. Stores the input sample and predicted value as metrics for tracking
  8. **Validate Monitoring** - Queries Prometheus (http://localhost:9090/metrics) to check that predict_requests_total metric exists.Queries Grafana API to verify the dashboard JSON is present.Fails the pipeline if either check fails
  9. Optionally clean up and notify

 ### Notes
  1. DVC Repro ensures any data preprocessing or intermediate files are updated.
  2. Train Model runs your train.py script.
  3. MLflow Logging stage assumes MLflow server is running; you can set MLFLOW_TRACKING_URI in Jenkins environment.
  4. Docker Build & Up automatically builds and launches your services (model API, MLflow, Grafana, Prometheus, etc.).
  5. Post stages handle success/failure notifications. You can extend them to email Slack notifications, etc.

---

## Notes

* Synthetic dataset is generated if `data/raw/housing.csv` is missing.
* `artifacts/` contains `model.joblib` and `metrics.json`.
* MLflow experiment `mlops-demo` is automatically created if it does not exist.

---

## What files are important
- `docker-compose.yml` — brings up Airflow, MLflow, Jenkins, Prometheus, Grafana, Model service
- `src/` — training & evaluation scripts
- `airflow_dags/train_dag.py` — DAG to orchestrate training
- `mlflow_server/` — Dockerfile to run MLflow tracking server
- `model_service/` — Flask app to serve model predictions and expose Prometheus metrics
- `dvc.yaml` — DVC pipeline referencing stages (preprocess, train, evaluate)


