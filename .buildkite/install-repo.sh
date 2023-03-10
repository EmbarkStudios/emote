set -eo pipefail

echo --- Setting up google-cloud-sdk

if [ -f '/gcloud/google-cloud-sdk/path.bash.inc' ]; then . '/gcloud/google-cloud-sdk/path.bash.inc'; fi
gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

echo --- Installing dependencies

pdm install -d -G ci
pdm torch install cpu
