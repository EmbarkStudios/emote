set -eo pipefail

echo --- Installing dependencies

poetry install --without gpu
pip install -r .buildkite/requirements.txt
