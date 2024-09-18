# hash:sha256:46f3173150b9242d413a3088fc246d29eb2240f1857f8341f4a54b7c3dc52c29
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    spikeinterface[full]==0.101.1 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
