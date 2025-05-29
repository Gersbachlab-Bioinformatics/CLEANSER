ARG PYTHON_VERSION=3.11.2-slim-bullseye

FROM python:${PYTHON_VERSION}

RUN apt-get update && apt-get install --no-install-recommends -qq wget ca-certificates make gcc g++ procps coreutils

WORKDIR /cleanser

COPY . .

ENV CMDSTAN_VERSION=2.36.0

RUN pip install --no-cache-dir build && \
    python -m build . && \
    pip install --no-cache-dir dist/*.whl && \
    install_cmdstan --version ${CMDSTAN_VERSION}

# Compile STAN models
RUN cd /root/.cmdstan/cmdstan-${CMDSTAN_VERSION} && \
    make /usr/local/lib/python3.11/site-packages/cleanser/cs-guide-mixture && \
    make /usr/local/lib/python3.11/site-packages/cleanser/dc-guide-mixture
