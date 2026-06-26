# hash:sha256:338893625319668e6db3e9c533f8f85a2348a3daf05f6a254cb14abb337d8cb6
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
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.104.7 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
