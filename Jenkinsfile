pipeline {
    agent any
    environment {
        DOCKER_COMPOSE_CMD = "docker-compose"
        DATA_DIR = "data"
        ARTIFACTS_DIR = "artifacts"
        MODEL_SERVICE_URL = "http://model-service:8000/predict"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
        VENV_PATH = "${WORKSPACE}/venv"
        PATH = "${VENV_PATH}/bin:${env.PATH}"
    }

    stages {

        // -------------------------
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/nithi-code/mlops-demo-scikit-learn-mlflow-airflow-docker.git',
                    credentialsId: 'github-pat'
            }
        }

        // -------------------------
        stage('Setup Python Environment') {
            steps {
                echo "Creating virtual environment and installing dependencies..."
                sh """
                    python3 -m venv ${VENV_PATH}
                    ${VENV_PATH}/bin/pip install --upgrade pip
                    ${VENV_PATH}/bin/pip install -r requirements.txt
                    ${VENV_PATH}/bin/pip install 'dvc[all]' mlflow requests
                """
            }
        }

        // -------------------------
        stage('Prepare Data / DVC') {
            steps {
                echo "Preparing data and DVC..."
                sh """
                    mkdir -p ${DATA_DIR}/processed ${ARTIFACTS_DIR}

                    # Only run DVC if repository exists
                    if [ -f "dvc.yaml" ] || [ -d ".dvc" ]; then
                        echo "DVC repo found. Attempting to pull data..."
                        set +e
                        ${VENV_PATH}/bin/dvc pull
                        if [ $? -ne 0 ]; then
                            echo "DVC pull failed or remote not configured. Using existing data."
                        fi
                        set -e
                    else
                        echo "No DVC repo found. Skipping DVC pull."
                    fi

                    # Verify processed CSV
                    if [ ! -f "${DATA_DIR}/processed/housing_processed.csv" ]; then
                        echo "[WARNING] Processed data not found. Training may fail!"
                    else
                        echo "Processed data is available."
                    fi
                """
            }
        }

        // -------------------------
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

        // -------------------------
        stage('Train Model') {
            steps {
                echo "Training the model..."
                sh "${VENV_PATH}/bin/python src/train.py"
            }
        }

        // -------------------------
        stage('Build & Deploy Docker Services') {
            steps {
                echo "Building and deploying Docker services..."
                sh "${DOCKER_COMPOSE_CMD} build"
                sh "${DOCKER_COMPOSE_CMD} up -d"
            }
        }

        // -------------------------
        stage('Test Model Prediction API') {
            steps {
                echo "Testing model prediction API..."
                script {
                    def payload = [[
                        "feature_0": 0.496714,
                        "feature_1": -0.138264,
                        "feature_2": 0.647688,
                        "feature_3": 1.52303,
                        "feature_4": -0.234153,
                        "feature_5": -0.234137,
                        "feature_6": 1.57921,
                        "feature_7": 0.767435
                    ]]
                    def payloadJson = groovy.json.JsonOutput.toJson(payload)
                    def response = sh(script: "curl -s -X POST -H 'Content-Type: application/json' -d '${payloadJson}' ${MODEL_SERVICE_URL}", returnStdout: true).trim()
                    echo "Prediction response: ${response}"
                }
            }
        }

        // -------------------------
        stage('Validate Monitoring') {
            steps {
                echo "Validating Prometheus metrics and Grafana dashboards..."
                script {
                    // Prometheus check
                    def prometheusResponse = sh(script: "curl -s http://prometheus:9090/metrics", returnStdout: true).trim()
                    if (!prometheusResponse.contains("predict_requests_total")) {
                        error "Prometheus metrics missing 'predict_requests_total'!"
                    }
                    echo "Prometheus metrics validation passed."

                    // Grafana dashboard check
                    def grafanaResponse = sh(script: "curl -s http://grafana:3000/api/dashboards/db/dashboard -u admin:admin", returnStdout: true).trim()
                    if (!grafanaResponse.contains("dashboard")) {
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
