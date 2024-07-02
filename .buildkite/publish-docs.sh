set -eo pipefail


source .buildkite/install-repo.sh

echo --- Building docs

if ! pants package //docs:book; then
	cat << EOF | buildkite-agent annotate --style "error" --context "pants"
:warning: Failed building documentation. Please check logs below, or build docs locally using \`pants package //docs:book\` to check for errors.
EOF
	exit 1
else
	if [[ "$BUILDKITE_BRANCH" = "main" ]]; then
		gsutil rsync -r ./dist/docs/book gs://embark-static/emote-docs
		buildkite-agent annotate "✅ New documentation deployed at https://static.embark.net/emote-docs/" --style "success" --context "pants"
	else
		buildkite-agent annotate "✅ Documentation built succesfully" --style "success" --context "pants"
	fi
fi
