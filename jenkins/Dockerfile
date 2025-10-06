# Base image
FROM ubuntu:22.04

# Switch to root for installing packages
USER root

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        curl \
        git \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate venv and upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip

# Install Python packages inside venv
RUN /opt/venv/bin/pip install dvc[all] mlflow requests

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Create Jenkins home directory
RUN mkdir -p /var/jenkins_home
VOLUME /var/jenkins_home

# Switch to Jenkins user
RUN useradd -m -d /var/jenkins_home jenkins
USER jenkins
WORKDIR /var/jenkins_home

# Expose Jenkins port
EXPOSE 8080

# Start Jenkins (the official Jenkins image CMD can be adapted)
CMD ["bash"]
