echo --- Initializing gcloud

if [ -f '/root/google-cloud-sdk/path.bash.inc' ]; then . '/root/google-cloud-sdk/path.bash.inc'; fi
gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

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

echo --- Building docs
pushd docs
make deploy
gsutil rsync -n -r ./_build/dirhtml gs://embark-static/emote-docs
popd
