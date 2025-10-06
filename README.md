
# MLOps Demo — scikit-learn (Docker Compose)

**Components included**
- DVC (dvc.yaml pipeline + .dvcignore + instructions)
- Airflow (DAG to run training)
- MLflow (tracking server Docker container)
- Jenkins (Jenkins image via Docker Compose)
- Prometheus + Grafana (monitoring)
- Docker Compose (orchestrates all services)
- Model service (Flask) exposing Prometheus metrics
- Sample dataset (synthetic)
- Scripts: preprocess, train, evaluate

This repo is intended for **Windows** (PowerShell) with Docker Desktop.

---

## Quick start (Windows PowerShell)

1. Install prerequisites:
   - Git
   - Docker Desktop (Windows)
   - Python 3.10+ (optional, for local runs)
   - DVC (optional; we include pipeline files you can run locally)

2. Unzip / clone the repo and open PowerShell in the project root.

3. Build and start all services:
```powershell
docker-compose up -d --build
```

Services and default ports:
- Airflow UI: http://localhost:8080  (user: `airflow` / `airflow`)
- MLflow: http://localhost:5000
- Jenkins: http://localhost:8081  (initial admin password in container logs)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000  (admin / admin)
- Model service (prediction + metrics): http://localhost:8000

4. Run the Airflow DAG `train_pipeline` from Airflow UI or wait for scheduled runs. The DAG executes the training script which logs experiments to MLflow and saves the model under `artifacts/`.

5. DVC:
```powershell
# Initialize locally (optional)
dvc init
dvc remote add -d localremote ./dvcstore
# To reproduce pipeline locally:
dvc repro
```

---

## What files are important
- `docker-compose.yml` — brings up Airflow, MLflow, Jenkins, Prometheus, Grafana, Model service
- `src/` — training & evaluation scripts
- `airflow_dags/train_dag.py` — DAG to orchestrate training
- `mlflow_server/` — Dockerfile to run MLflow tracking server
- `model_service/` — Flask app to serve model predictions and expose Prometheus metrics
- `dvc.yaml` — DVC pipeline referencing stages (preprocess, train, evaluate)

---

## Notes
- This demo uses scikit-learn; model is lightweight.
- The Compose setup uses official images. First `docker-compose up --build` may download several images.
- On Windows, ensure Docker Desktop has enough resources (CPU, RAM) — adjust if necessary.
