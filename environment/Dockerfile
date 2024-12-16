# hash:sha256:76e0cc4a9259c0f72a1da8778c8aa256d4f69c4f5a538d2c5ccc6c85417ca63d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.1 \
    spikeinterface[full]==0.101.2 \
    torch==2.2.0 \
    wavpack-numcodecs==0.2.1
