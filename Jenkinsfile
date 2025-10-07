pipeline {
    agent any
    environment {
        DOCKER_COMPOSE_CMD = "docker-compose"
        DATA_DIR = "data"
        ARTIFACTS_DIR = "artifacts"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
        MLFLOW_ARTIFACTS_DIR = "${WORKSPACE}/mlflow_artifacts"
        VENV_PATH = "${WORKSPACE}/venv"
        PATH = "${VENV_PATH}/bin:${env.PATH}"
        PROCESSED_PATH = "${DATA_DIR}/processed/housing_processed.csv"
        PROMETHEUS_HOST = "prometheus"
        GRAFANA_HOST = "grafana"
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
                echo "Creating virtual environment and installing dependencies..."
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
                    ${VENV_PATH}/bin/dvc pull || true
                """
            }
        }

        stage('Preprocess Data') {
            steps {
                echo "Running preprocessing if processed data is missing..."
                sh """
                    if [ ! -f "${PROCESSED_PATH}" ]; then
                        echo "Processed data not found. Running preprocess.py..."
                        ${VENV_PATH}/bin/python src/preprocess.py
                    else
                        echo "Processed data exists. Skipping preprocessing."
                    fi
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
                sh """
                    mkdir -p ${WORKSPACE}/mlflow_artifacts
                    export PROCESSED_PATH=${PROCESSED_PATH}
                    export MLFLOW_ARTIFACTS_DIR=${WORKSPACE}/mlflow_artifacts
                    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
                    ${VENV_PATH}/bin/python src/train.py
                """
            }
        }

        // stage('Build & Deploy Services') {
        //     steps {
        //         echo "Building and deploying Docker services..."
        //         sh "${DOCKER_COMPOSE_CMD} build"
        //         sh "${DOCKER_COMPOSE_CMD} up -d"
        //     }
        // }

        stage('Test Model Prediction') {
            steps {
                echo "Testing model prediction API..."
                sh """
                    # Create payload JSON file
                    cat > payload.json <<EOF
        [
        {
            "feature_0": 0.496714,
            "feature_1": -0.138264,
            "feature_2": 0.647688,
            "feature_3": 1.52303,
            "feature_4": -0.234153,
            "feature_5": -0.234137,
            "feature_6": 1.57921,
            "feature_7": 0.767435
        }
        ]
        EOF

                    # Call prediction API
                    RESPONSE=\$(curl -s -X POST -H 'Content-Type: application/json' -d @payload.json http://localhost:8000/predict)
                    echo "Prediction response: \$RESPONSE"
                """
            }
        }


        stage('Validate Monitoring') {
            steps {
                echo "Validating Prometheus metrics and Grafana dashboards..."
                sh """
                    # Wait for Prometheus
                    until curl -s http://${PROMETHEUS_HOST}:9090/metrics >/dev/null; do
                        echo "Waiting for Prometheus..."
                        sleep 5
                    done
                    PROM_RESPONSE=\$(curl -s http://${PROMETHEUS_HOST}:9090/metrics)
                    if ! echo "\$PROM_RESPONSE" | grep -q "predict_requests_total"; then
                        echo "Prometheus metrics missing 'predict_requests_total'!"
                        exit 1
                    fi
                    echo "Prometheus metrics validation passed."

                    # Wait for Grafana
                    until curl -s http://${GRAFANA_HOST}:3000/api/dashboards/db/dashboard >/dev/null; do
                        echo "Waiting for Grafana..."
                        sleep 5
                    done
                    GRAF_RESPONSE=\$(curl -s http://${GRAFANA_HOST}:3000/api/dashboards/db/dashboard)
                    if ! echo "\$GRAF_RESPONSE" | grep -q "dashboard"; then
                        echo "Grafana dashboard validation failed!"
                        exit 1
                    fi
                    echo "Grafana dashboard validation passed."
                """
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
