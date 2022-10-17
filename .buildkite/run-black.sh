set -eo pipefail

source .buildkite/install-repo.sh

echo --- Running black

EXIT_CODE=0
pdm run black --check --diff emote > diff.txt || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "eslint"
:warning: Your code isn't formatted by \`black\`. Please fix the below diffs, or run \`pdm run black emote\` to automatically format it.

\`\`\`diff
$(cat diff.txt)
\`\`\`

EOF
else
	buildkite-agent annotate "✅ Code formatted correctly " --style "success" --context "eslint"
fi

exit $EXIT_CODE
