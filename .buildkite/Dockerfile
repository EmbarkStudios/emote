FROM docker.io/buildpack-deps:22.04-curl

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100 \
	PATH="$PATH:/root/.local/bin/:/.pyenv/bin/" \
	PYENV_ROOT="/.pyenv"

RUN apt-get -y update && \
    apt-get -y install curl build-essential gcc git swig libffi-dev libncurses5-dev zlib1g zlib1g-dev libssl-dev libsqlite3-dev liblzma-dev libreadline-dev libbz2-dev && \
    apt-get -y clean

RUN curl https://pyenv.run | bash && \
	eval "$(pyenv init -)" && \
	pyenv install 3.11 && \
	pyenv global 3.11 && \
	pyenv install 3.10 && \
	pyenv install 3.9 && \
	pyenv install 3.8 && \
	pyenv rehash && \
	python --version && \
	pyenv exec pip --version

RUN mkdir -p /gcloud/ && \
	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-linux-x86_64.tar.gz && \
	tar -xf google-cloud-cli-408.0.1-linux-x86_64.tar.gz -C /gcloud/ && \
	/gcloud/google-cloud-sdk/install.sh --quiet --usage-reporting false && \
	rm google-cloud-cli-408.0.1-linux-x86_64.tar.gz

RUN eval "$(pyenv init --path)" && \
	eval "$(pyenv init -)" && \
	pyenv global 3.11 && python --version && python -m pip install \
	--no-cache-dir pipx && \
	pyenv rehash && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.3.0 --suffix 23 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.4.0 --suffix 24 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.5.0 --suffix 25 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.6.0 --suffix 26 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.7.0 --suffix 27 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.8.0 --suffix 28 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.9.0 --suffix 29 && \
	pipx install --pip-args='--no-cache-dir' pdm~=2.10.0 --suffix 210 && \
	pdm23 --version && \
	pdm24 --version && \
	pdm25 --version && \
	pdm26 --version && \
	pdm27 --version && \
	pdm28 --version && \
	pdm29 --version && \
	pdm210 --version

ENTRYPOINT bash
