# hash:sha256:a06de692574a376b843594f2812de0bbe5814f29f62323612ddf391a2c0b344f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.19.1 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
