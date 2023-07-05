set -eo pipefail


source .buildkite/install-repo.sh

echo --- Building docs
pushd docs
EXIT_CODE=0
PDM=${PDM_COMMAND:1:-1} ${PDM_COMMAND:1:-1} run make deploy  || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "sphinx"
:warning: Failed building documentation. Please check logs below, or build docs locally using `make deploy` to check for errors.
EOF
	exit 1
else
	if [[ "$BUILDKITE_BRANCH" = "main" ]]; then
		gsutil rsync -r ./_build/dirhtml gs://embark-static/emote-docs
		buildkite-agent annotate "✅ New documentation deployed at https://static.embark.net/emote-docs/" --style "success" --context "sphinx"
	else
		buildkite-agent annotate "✅ Documentation built succesfully" --style "success" --context "sphinx"
	fi
fi
popd
