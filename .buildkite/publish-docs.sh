set -eo pipefail


source .buildkite/install-repo.sh

echo --- Initializing gcloud

curl https://sdk.cloud.google.com > install.sh && \
    bash install.sh --disable-prompts 2>&1 && \
    /root/google-cloud-sdk/install.sh --path-update true --usage-reporting false --quiet

if [ -f '/root/google-cloud-sdk/path.bash.inc' ]; then . '/root/google-cloud-sdk/path.bash.inc'; fi
gcloud config set account monorepo-ci@embark-builds.iam.gserviceaccount.com

echo --- Building docs
pushd docs
EXIT_CODE=0
make deploy || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "sphinx"
:warning: Failed building documentation. Please check logs below, or build docs locally using `make deploy` to check for errors.
EOF
else
	if [[ "$BUILDKITE_BRANCH" = "main" ]]; then
		pdm run gsutil rsync -r ./_build/dirhtml gs://embark-static/emote-docs
		buildkite-agent annotate "✅ New documentation deployed at https://static.embark.net/emote-docs/" --style "success" --context "sphinx"
	else
		buildkite-agent annotate "✅ Documentation built succesfully" --style "success" --context "sphinx"
	fi
fi
popd
