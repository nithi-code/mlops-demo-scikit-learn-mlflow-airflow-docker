pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
            args '-v /var/jenkins_home:/var/jenkins_home'
        }
    }
    environment {
        DOCKER_COMPOSE_CMD = "docker-compose"
        DATA_DIR = "data"
        ARTIFACTS_DIR = "artifacts"
        MODEL_SERVICE_URL = "http://model-service:8000/predict"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
    }
    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main',
                url: 'https://github.com/nithi-code/mlops-demo-scikit-learn-mlflow-airflow-docker.git',
                credentialsId: 'github-pat'
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Installing Python dependencies..."
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                sh 'pip install dvc[all] mlflow requests'
            }
        }

        stage('DVC Repro') {
            steps {
                echo "Reproducing DVC pipeline..."
                sh 'dvc repro || true'
            }
        }

        stage('Train Model') {
            steps {
                echo "Training the model..."
                sh 'python src/train.py'
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
                    def payload = [[
                        "feature_0": 0.496714, "feature_1": -0.138264, "feature_2": 0.647688,
                        "feature_3": 1.52303, "feature_4": -0.234153, "feature_5": -0.234137,
                        "feature_6": 1.57921, "feature_7": 0.767435
                    ]]
                    def payloadJson = groovy.json.JsonOutput.toJson(payload)
                    def response = sh(
                        script: "curl -s -X POST -H 'Content-Type: application/json' -d '${payloadJson}' ${MODEL_SERVICE_URL}",
                        returnStdout: true
                    ).trim()
                    echo "Prediction response: ${response}"

                    // Log prediction to MLflow
                    sh """
                    python - <<EOF
import mlflow, json
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
                    // Prometheus
                    def prometheusResponse = sh(script: "curl -s http://localhost:9090/metrics", returnStdout: true).trim()
                    if (!prometheusResponse.contains("predict_requests_total")) {
                        error "Prometheus metrics missing 'predict_requests_total'!"
                    }
                    echo "Prometheus metrics validation passed."

                    // Grafana
                    def grafanaResponse = sh(script: "curl -s -u admin:admin http://localhost:3000/api/dashboards/db/dashboard", returnStdout: true).trim()
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
