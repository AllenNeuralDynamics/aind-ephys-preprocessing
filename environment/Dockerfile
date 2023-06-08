# hash:sha256:3b486d4f3eb81167665f2c0fbd2f2e5df8cce8cf001219bbc141d867b9989d0a
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.1.5 \
    wavpack-numcodecs==0.1.3

COPY postInstall /
RUN /postInstall
