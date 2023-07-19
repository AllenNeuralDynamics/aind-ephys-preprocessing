# hash:sha256:e7fa75b5f7ab0065b9eaa8005e8a6c0d4e8eeb3cc319ae71cc9d3c38c9debc56
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.13.61 \
    wavpack-numcodecs==0.1.4

COPY postInstall /
RUN /postInstall
