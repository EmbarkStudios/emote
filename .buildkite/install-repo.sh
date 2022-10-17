set -eo pipefail

echo --- Installing dependencies

pdm install -d -g ci
