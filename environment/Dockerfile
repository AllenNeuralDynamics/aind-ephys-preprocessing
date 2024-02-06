# hash:sha256:1634206687c30dc5f42e113c68c8f36a1ccb359d54cd16f7916987e03acb67a9
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    spikeinterface[full]==0.100.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
