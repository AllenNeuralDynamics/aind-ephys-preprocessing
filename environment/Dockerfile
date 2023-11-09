# hash:sha256:7ee64c4597beee1bb8e55e9deb4443593888cc52b54a17a0349f0c9525f2c719
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25 \
    wavpack-numcodecs==0.1.5
