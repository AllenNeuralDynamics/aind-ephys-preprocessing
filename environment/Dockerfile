# hash:sha256:fbb9c2d640b89715c80c7f1c88adcd9a282a9a3303f3a42d7fbbd8def12a8dfa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.5 \
    spikeinterface[full]==0.101.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
