# hash:sha256:3e705815755f6ac989838971875e8f7fdbcd7b92b70a1e114a4c592dbe67af72
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.13.50 \
    wavpack-numcodecs==0.1.4

COPY postInstall /
RUN /postInstall
