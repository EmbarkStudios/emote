set -eo pipefail

${PDM_COMMAND:1:-1} install --plugins

EXIT_CODE=0
TORCH_EXIT_CODE=0

${PDM_COMMAND:1:-1} lock --check || EXIT_CODE=$?
${PDM_COMMAND:1:-1} torch lock --check || TORCH_EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ $TORCH_EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "lockfile"
:lock: Lockfiles are outdated. Please run \`pdm lock && pdm torch lock\` and commit the result.
EOF
	exit 1
else
    buildkite-agent annotate --style "success" --context "lockfile" ":lock: Lockfile is up to date."
	exit 0
fi
