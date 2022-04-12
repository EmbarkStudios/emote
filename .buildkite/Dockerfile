FROM docker.io/python@sha256:50e21e0bac13e1dfa37626d1c05433cc29e0f1d15fa390e2ecbae32221c6646d

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.0b1 \
    POETRY_HOME="/tmp/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get -y update && \
    apt-get -y install curl build-essential gcc git && \
    apt-get -y clean

RUN pip install \
	--no-cache-dir \
	poetry==1.2.0b1

ENTRYPOINT bash
