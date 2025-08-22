# docker/Dockerfile.lasso
FROM mambaorg/micromamba:1.5.10 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
SHELL ["/bin/bash","-lc"]

# Conda env
COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba clean -a -y

# ---- Production image (for AWS Batch) ----
FROM base AS prod
# Copy project (adjust context path as needed)
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
USER $MAMBA_USER
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["bash"]

# ---- Dev image (extra tools for local use) ----
FROM base AS dev
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      vim less git curl procps tzdata ca-certificates \
  && rm -rf /var/lib/apt/lists/*
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
USER $MAMBA_USER
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["bash"]


