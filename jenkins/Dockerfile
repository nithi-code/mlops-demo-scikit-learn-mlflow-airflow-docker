# Base image: official Jenkins LTS
FROM jenkins/jenkins:lts

USER root
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

# Create and configure Python venv
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install dvc[all] mlflow requests

ENV PATH="/opt/venv/bin:$PATH"

# Allow Jenkins user to use Docker safely
# Replace 999 with your host Docker group GID (`ls -l /var/run/docker.sock`)
ARG DOCKER_GID=999
RUN groupadd -g $DOCKER_GID docker || true && \
    usermod -aG docker jenkins && \
    mkdir -p /var/jenkins_home && chown -R jenkins:jenkins /var/jenkins_home

USER jenkins
WORKDIR /var/jenkins_home

# Jenkins web interface
EXPOSE 8080

# Correct entrypoint for Jenkins
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/jenkins.sh"]
