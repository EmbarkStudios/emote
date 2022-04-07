set -eo pipefail

echo --- Installing dependencies

apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential

export PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.0b1 \
    POETRY_HOME="/tmp/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \

pip install poetry==1.2.0b1
poetry install
poetry env info --path

echo --- Initializing gcloud

curl https://sdk.cloud.google.com > install.sh && \
    bash install.sh --disable-prompts 2>&1 && \
    /root/google-cloud-sdk/install.sh --path-update true --usage-reporting false --quiet

if [ -f '/root/google-cloud-sdk/path.bash.inc' ]; then . '/root/google-cloud-sdk/path.bash.inc'; fi
gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

echo --- Building docs
pushd docs
poetry env info --path
EXIT_CODE=0
make deploy || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "sphinx"
:warning: Failed building documentation. Please check logs below, or build docs locally using `make deploy` to check for errors.
EOF
else
	if [[ "$BUILDKITE_BRANCH" = "main" ]]; then
		gsutil rsync -r ./_build/dirhtml gs://embark-static/emote-docs
		buildkite-agent annotate "✅ New documentation deployed at https://static.embark.net/emote-docs/" --style "success" --context "sphinx"
	else
		buildkite-agent annotate "✅ Documentation built succesfully" --style "success" --context "sphinx"
	fi
fi
popd
