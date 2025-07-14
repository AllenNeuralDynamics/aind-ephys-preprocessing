# hash:sha256:955278be844927d109940426f83831937bf4ce1a01ad6ece7949d6e75f786a68
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.102.3 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
