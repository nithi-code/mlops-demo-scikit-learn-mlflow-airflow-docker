pipeline {
    agent any
    environment {
        DOCKER_COMPOSE_CMD = "docker-compose"
        DATA_DIR = "data"
        ARTIFACTS_DIR = "artifacts"
        MODEL_SERVICE_URL = "http://localhost:8000/predict"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
        VENV_PATH = "${WORKSPACE}/venv"
        PATH = "${VENV_PATH}/bin:${env.PATH}"
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/nithi-code/mlops-demo-scikit-learn-mlflow-airflow-docker.git',
                    credentialsId: 'github-pat'
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Creating virtual environment and installing Python dependencies..."
                sh """
                    python3 -m venv ${VENV_PATH}
                    ${VENV_PATH}/bin/pip install --upgrade pip
                    ${VENV_PATH}/bin/pip install -r requirements.txt
                    ${VENV_PATH}/bin/pip install 'dvc[all]' mlflow requests
                """
            }
        }

        stage('DVC Pull') {
            steps {
                echo "Pulling data from DVC..."
                sh """
                    mkdir -p ${DATA_DIR}/processed ${ARTIFACTS_DIR}
                    ${VENV_PATH}/bin/dvc pull
                """
            }
        }

        stage('Wait for MLflow') {
            steps {
                echo "Waiting for MLflow Tracking Server..."
                sh """
                    until curl -s ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list >/dev/null; do
                        echo "Waiting for MLflow..."
                        sleep 5
                    done
                """
            }
        }

        stage('Train Model') {
            steps {
                echo "Training the model..."
                sh "${VENV_PATH}/bin/python src/train.py"
            }
        }

        stage('Build & Deploy Services') {
            steps {
                echo "Building and deploying Docker services..."
                sh "${DOCKER_COMPOSE_CMD} build"
                sh "${DOCKER_COMPOSE_CMD} up -d"
            }
        }

        stage('Test Model Prediction') {
            steps {
                echo "Testing model prediction API..."
                script {
                    def payload = [
                        [
                            "feature_0": 0.496714,
                            "feature_1": -0.138264,
                            "feature_2": 0.647688,
                            "feature_3": 1.52303,
                            "feature_4": -0.234153,
                            "feature_5": -0.234137,
                            "feature_6": 1.57921,
                            "feature_7": 0.767435
                        ]
                    ]
                    def payloadJson = groovy.json.JsonOutput.toJson(payload)
                    def response = sh(script: "curl -s -X POST -H 'Content-Type: application/json' -d '${payloadJson}' ${MODEL_SERVICE_URL}", returnStdout: true).trim()
                    echo "Prediction response: ${response}"

                    // Log prediction to MLflow
                    sh """${VENV_PATH}/bin/python - <<EOF
import mlflow
import json
mlflow.set_tracking_uri("${MLFLOW_TRACKING_URI}")
with mlflow.start_run(run_name="test_prediction"):
    mlflow.log_param("input_sample", '${payloadJson}')
    mlflow.log_metric("predicted_value", json.loads('${response}')['predictions'][0])
EOF
                    """
                }
            }
        }

        stage('Validate Monitoring') {
            steps {
                echo "Validating Prometheus metrics and Grafana dashboards..."
                script {
                    def prometheusResponse = sh(script: "curl -s http://localhost:9090/metrics", returnStdout: true).trim()
                    if (!prometheusResponse.contains("predict_requests_total")) {
                        error "Prometheus metrics missing 'predict_requests_total'!"
                    }
                    echo "Prometheus metrics validation passed."

                    def grafanaDashboardResponse = sh(script: "curl -s http://localhost:3000/api/dashboards/db/dashboard", returnStdout: true).trim()
                    if (!grafanaDashboardResponse.contains("dashboard")) {
                        error "Grafana dashboard validation failed!"
                    }
                    echo "Grafana dashboard validation passed."
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs for details."
        }
    }
}
