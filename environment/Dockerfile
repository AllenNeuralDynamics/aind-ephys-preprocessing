# hash:sha256:b944eb213b9af359054cafbc61cb33f6a4f6b3d7a5aa7322de7bc3389c5fbfa1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    log-schema==0.2 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.105.0 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
