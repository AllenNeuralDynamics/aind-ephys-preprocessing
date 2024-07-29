# hash:sha256:98e5fdc8aa623cecfc15fb556e65bdc6ffec81c61fe01870b824f52d9698e7fe
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    spikeinterface[full]==0.101.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
