# docker/Dockerfile.lasso

FROM mambaorg/micromamba:1.5.10 AS base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app
SHELL ["/bin/bash","-lc"]
USER root

# Env + awscli
COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba install -y -n base -c conda-forge awscli \
 && micromamba clean -a -y \
 && ln -sf /opt/conda/bin/python  /usr/local/bin/python3 \
 && ln -sf /opt/conda/bin/mafft   /usr/local/bin/mafft \
 && ln -sf /opt/conda/bin/aws     /usr/local/bin/aws

# ---- Production image (for AWS Batch) ----
FROM base AS prod
# Copy your updated sources, including scripts/preprocess_all.py
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
USER $MAMBA_USER

# IMPORTANT: no ENTRYPOINT that cd's or swallows Nextflow's wrapper
CMD ["bash"]
