# hash:sha256:b9dfde6c222f38b39f46b076319b828a5ad39b949cf034625bbfcf0d8c855e2e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25 \
    spikeinterface[full]==0.99.0 \
    wavpack-numcodecs==0.1.5
