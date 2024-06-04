# hash:sha256:e1c3f3db319b6c8b7c326f1c131ca80c125f137b0bbddaae85dbf1a1cec36f6a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
