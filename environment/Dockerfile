# hash:sha256:8d7253b4ec95b8cf0ae25eece50a9eaa75c133227ff25b381171d89603ffb4f5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib \
    numpy \
    pandas \
    scipy
