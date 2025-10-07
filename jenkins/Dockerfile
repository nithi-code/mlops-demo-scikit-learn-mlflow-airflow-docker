# Base image: official Jenkins LTS
FROM jenkins/jenkins:lts

# Switch to root to install dependencies
USER root

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        curl \
        git \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Activate venv and upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip

# Install Python packages inside venv
RUN /opt/venv/bin/pip install dvc[all] mlflow requests

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Add Jenkins user to docker group
RUN usermod -aG docker jenkins

# Ensure Jenkins home has correct permissions
RUN mkdir -p /var/jenkins_home && chown -R jenkins:jenkins /var/jenkins_home

# Switch back to Jenkins user
USER jenkins
WORKDIR /var/jenkins_home

# Expose Jenkins default port
EXPOSE 8080

# Use official Jenkins startup script
CMD ["/sbin/tini", "--", "/usr/local/bin/jenkins.sh"]
