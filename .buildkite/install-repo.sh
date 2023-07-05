set -eo pipefail

echo --- Setting up google-cloud-sdk

if [ -f '/gcloud/google-cloud-sdk/path.bash.inc' ]; then . '/gcloud/google-cloud-sdk/path.bash.inc'; fi
gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

echo --- Installing dependencies

${PDM_COMMAND:1:-1} install -d -G ci -k post_install
${PDM_COMMAND:1:-1} torch install cpu
