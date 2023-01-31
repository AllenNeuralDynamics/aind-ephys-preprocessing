# hash:sha256:6b1259d8167940881d719b0e303cc546ebb3019a6d665ecab0338fb60c8281e7
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.1.5 \
    neo==0.11.1 \
    probeinterface==0.2.15 \
    wavpack-numcodecs==0.1.3

COPY postInstall /
RUN /postInstall
