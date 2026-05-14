# hash:sha256:3db11c282f847f9d00927299636bd4a0527b2663f19cc0da15d43244d368679f
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
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    scipy==1.17.1 \
    spikeinterface[full]==0.103.2 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
