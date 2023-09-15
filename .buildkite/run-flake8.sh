set -eo pipefail

source .buildkite/install-repo.sh

echo --- Running flake8

EXIT_CODE=0
${PDM_COMMAND:1:-1} run flake8 emote experiments tests > diff.txt || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "flake8"
:warning: \`flake8\` found issues with your code. Please fix the below issues, and update your PR.

\`\`\`diff
$(cat diff.txt)
\`\`\`

EOF
else
	buildkite-agent annotate "✅ \`flake8\` found no code issues." --style "success" --context "flake8"
fi

exit $EXIT_CODE
