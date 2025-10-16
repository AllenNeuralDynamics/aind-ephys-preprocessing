# hash:sha256:f210e2562d533851b99272e809ea5d44796de4b1b2a8ec7e8c3a8efdc4fa5916
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.0 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
