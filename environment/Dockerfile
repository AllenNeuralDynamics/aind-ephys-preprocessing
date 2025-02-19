# hash:sha256:5ce86b8c8b6a03c200d182a8138c8cc5c5b93eddf7da69fbfdac4c4893796612
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.102.1 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
