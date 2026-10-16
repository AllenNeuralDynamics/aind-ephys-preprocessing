# hash:sha256:2dc469098d08fde33ec33081295384cd0d25fc3c3a9aef1eec71360cf2b147df
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
    log-schema==0.2.10.dev1 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.105.2 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
