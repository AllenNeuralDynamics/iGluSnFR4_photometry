# hash:sha256:1fdbe0f98b00d42feac3b145a148e83f81aea3199c8f56ab24de8b653e0e7c8e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.3.0 \
    scikit-posthocs==0.11.4 \
    scipy==1.13.1
