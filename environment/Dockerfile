# hash:sha256:e4009168a6b533e50e5c10d979724f54178b062e18819eef9ff577b2d14145a7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.101.2 \
    torch==2.2.0 \
    wavpack-numcodecs==0.2.1
