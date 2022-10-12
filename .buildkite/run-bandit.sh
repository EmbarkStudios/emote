set -eo pipefail

source .buildkite/install-repo.sh

echo --- Running bandit

EXIT_CODE=0
poetry run bandit --r emote experiments tests -ll > diff.txt || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "bandit"
:warning: \`bandit\` found issues with your code. Please fix the below, and update your PR.

\`\`\`diff
$(cat diff.txt)
\`\`\`

EOF
else
	buildkite-agent annotate "✅ \`bandit\` found no code issues." --style "success" --context "eslint"
fi

exit $EXIT_CODE