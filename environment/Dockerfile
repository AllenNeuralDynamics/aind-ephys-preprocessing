# hash:sha256:ab9235403c68296fd6df7a9aad809fab29baeb4b34a9632b8f4ccb693da7b68d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    spikeinterface[full]==0.100.7 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
