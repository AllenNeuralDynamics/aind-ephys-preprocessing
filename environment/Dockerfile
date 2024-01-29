# hash:sha256:f4ff222d37078971c6d0d7acb2860995b933f1eaadb20cc4529ad91b73775dae
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
