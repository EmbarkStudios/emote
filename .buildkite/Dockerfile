FROM docker.io/python@sha256:832385dfbafed97b56379fa2665f858d9178f8fea4bc6efc8d3282804d68a6b4

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100 \
	PATH="$PATH:/root/.local/bin/"

RUN apt-get -y update && \
    apt-get -y install curl build-essential gcc git swig && \
    apt-get -y clean

RUN mkdir -p /gcloud/ && \
	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-linux-x86_64.tar.gz && \
	tar -xf google-cloud-cli-408.0.1-linux-x86_64.tar.gz -C /gcloud/ && \
	/gcloud/google-cloud-sdk/install.sh --quiet --usage-reporting false && \
	rm google-cloud-cli-408.0.1-linux-x86_64.tar.gz

RUN pip install \
	--no-cache-dir \
	pipx && \
	pipx install pdm~=2.4.0 && \
	pipx install pdm~=2.3.0 --suffix 23 && \
	pipx install pdm~=2.4.0 --suffix 24 && \
	pipx install pdm~=2.5.0 --suffix 25 && \
	pdm23 --version && \
	pdm24 --version && \
	pdm25 --version && \
	pdm --version

ENTRYPOINT bash
