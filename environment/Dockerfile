# hash:sha256:ff42bee5f8a96bb5650c8db6c0fbba76c0c21c34e05da560b31764df492b59ac
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.1.5 \
    neo==0.11.1 \
    probeinterface==0.2.16 \
    wavpack-numcodecs==0.1.3

COPY postInstall /
RUN /postInstall
