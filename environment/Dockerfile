# hash:sha256:b82588984245c160435f2f3838a267a618a67231df95478f4ef3fb664ed79c68
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    spikeinterface[full]==0.101.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
