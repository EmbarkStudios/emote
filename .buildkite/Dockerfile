FROM docker.io/python@sha256:50e21e0bac13e1dfa37626d1c05433cc29e0f1d15fa390e2ecbae32221c6646d

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get -y update && \
    apt-get -y install curl build-essential gcc git && \
    apt-get -y clean

RUN mkdir -p /gcloud/ && \
	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-linux-x86_64.tar.gz && \
	tar -xf google-cloud-cli-408.0.1-linux-x86_64.tar.gz -C /gcloud/ && \
	/gcloud/google-cloud-sdk/install.sh --quiet --usage-reporting false && \
	rm google-cloud-cli-408.0.1-linux-x86_64.tar.gz

RUN pip install \
	--no-cache-dir \
	pdm==2.2.0

ENTRYPOINT bash
