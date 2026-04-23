# hash:sha256:b3aad2addcb9f41f62c60b09c0e36c1aae5f942bba028e43bcc319fd5ac391c4
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
    spikeinterface[full]==0.104.3 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
