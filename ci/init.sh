#!/usr/bin/env bash

set -euxo pipefail

if [[ ! -v "GITHUB_ACTIONS" ]]; then
	apt update
	apt install -y build-essential \
		libssl-dev \
		zlib1g-dev \
		libbz2-dev \
		libreadline-dev \
		libsqlite3-dev \
		curl \
		libncursesw5-dev \
		xz-utils \
		tk-dev \
		libxml2-dev \
		libxmlsec1-dev \
		libffi-dev \
		liblzma-dev \
		swig \
		graphviz
fi

export PATH="/root/bin/:$PATH"

if ! command -v pants &>/dev/null; then
	bash get-pants.sh
fi

gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

echo "--- Running main build step"
