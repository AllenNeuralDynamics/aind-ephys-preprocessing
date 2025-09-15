# hash:sha256:3c2301421ff24b4fd8f855980db0a89ca5c9d9efe5cee6cfdbc39102faf4adbb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.1 \
    aind-data-schema-models==4.2.10 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.0 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
