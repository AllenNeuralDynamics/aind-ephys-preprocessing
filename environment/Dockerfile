# hash:sha256:ad8fde621c278f892ad61328d62449fc94f332af2832cc97c67dd61a4c8299c5
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
    wavpack-numcodecs==0.2.1

COPY postInstall /
RUN /postInstall
