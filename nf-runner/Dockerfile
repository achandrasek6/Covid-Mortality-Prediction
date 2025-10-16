# Runner image for launching Nextflow on AWS Batch (submits worker jobs)
# - Minimal: Java + Nextflow + AWS CLI + bash utils
# - Does NOT include your bio tools (those live in the step containers)

FROM debian:bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG NXF_VER=24.10.1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl unzip git \
      openjdk-17-jre-headless locales \
      tini less procps coreutils \
  && rm -rf /var/lib/apt/lists/*

# Locale (avoid UTF-8 issues)
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# AWS CLI v2
RUN curl -L "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
  && unzip -q /tmp/awscliv2.zip -d /tmp \
  && /tmp/aws/install \
  && rm -rf /tmp/aws /tmp/awscliv2.zip

# Non-root user
RUN useradd -m -u 10001 -s /bin/bash appuser
USER appuser

# Nextflow dirs and PATH
ENV NXF_HOME=/home/appuser/.nextflow \
    NXF_ASSETS=/home/appuser/.nextflow/assets \
    PATH=/home/appuser/bin:$PATH \
    NXF_TEMP=/tmp \
    TMPDIR=/tmp \
    HOME=/home/appuser

# Install Nextflow (pin via NXF_VER build-arg) and pre-seed the one-jar
RUN mkdir -p /home/appuser/bin \
  && curl -fsSL "https://get.nextflow.io" -o /home/appuser/bin/nextflow \
  && chmod +x /home/appuser/bin/nextflow \
  && mkdir -p "${NXF_HOME}/framework/${NXF_VER}" \
  && curl -fsSL "https://www.nextflow.io/releases/v${NXF_VER}/nextflow-${NXF_VER}-one.jar" \
       -o "${NXF_HOME}/framework/${NXF_VER}/nextflow-${NXF_VER}-one.jar" \
  && /home/appuser/bin/nextflow -version


# Workdir for transient logs (pipeline writes to S3)
WORKDIR /workspace

# Copy pipeline files from repo root (build with context at repo root)
COPY --chown=appuser:appuser main.nf /pipeline/main.nf
COPY --chown=appuser:appuser nextflow.config /pipeline/nextflow.config

# Copy entrypoint from nf-runner/
COPY --chown=appuser:appuser nf-runner/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of the pipeline that Nextflow stages into task work dirs
COPY --chown=appuser:appuser scripts/   /pipeline/scripts/
COPY --chown=appuser:appuser model_artifacts/      /pipeline/model_artifacts/
COPY --chown=appuser:appuser lasso_training_data/  /pipeline/lasso_training_data/
COPY --chown=appuser:appuser raw_data/  /pipeline/raw_data/

# Tini as PID 1 for clean signals
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD []
