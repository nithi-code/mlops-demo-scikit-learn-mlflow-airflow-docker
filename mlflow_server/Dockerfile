FROM python:3.10-slim

WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow[extras]

# Ensure artifacts folder exists
RUN mkdir -p /mlflow_artifacts && chmod -R 777 /mlflow_artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow_artifacts", "--serve-artifacts"]
