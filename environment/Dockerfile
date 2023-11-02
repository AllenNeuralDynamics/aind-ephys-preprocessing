# hash:sha256:58bdfbb90c9d122dc933ac36077ac3839609d8545fc37715334dd22e15624d54
FROM registry.codeocean.allenneuraldynamics-test.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.13.66 \
    wavpack-numcodecs==0.1.4

COPY postInstall /
RUN /postInstall
