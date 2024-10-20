# hash:sha256:e3e9f0911a8daee1fb0fcbda61985551df86c80b7765417f3b8f4b2f0c602967
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    spikeinterface[full]==0.101.2 \
    torch==2.2.0 \
    wavpack-numcodecs==0.2.1
