# hash:sha256:87419e0ef9b324dc6ccf944fa438e518476e1511c22eabc40c461e4f7dea4621
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

COPY postInstall /
RUN /postInstall
