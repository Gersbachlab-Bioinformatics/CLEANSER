ARG PYTHON_VERSION=3.11.2-slim-bullseye

FROM python:${PYTHON_VERSION}

RUN apt-get update && apt-get install --no-install-recommends -qq wget ca-certificates make gcc g++ procps coreutils

WORKDIR /cleanser

COPY . .

RUN pip install build
RUN python -m build .
RUN pip install dist/*.whl
RUN install_cmdstan

