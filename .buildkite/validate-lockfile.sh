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

EXIT_CODE=0
poetry lock --check || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    buildkite-agent annotate --style "error" --context "lockfile" ":lock: Lockfile is outdated. Please run \`poetry lock --no-update\` and commit the result."
  exit 1
fi
