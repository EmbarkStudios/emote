set -eo pipefail

ls -alh /root/.local/bin/
echo $(which pdm25)
echo $PATH
$PDM_COMMAND plugin add pdm-plugin-torch==23.1

EXIT_CODE=0
TORCH_EXIT_CODE=0

$PDM_COMMAND lock --check || EXIT_CODE=$?
$PDM_COMMAND torch lock --check || TORCH_EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ $TORCH_EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "lockfile"
:lock: Lockfiles are outdated. Please run \`pdm lock\` and commit the result.
EOF
	exit 1
else
    buildkite-agent annotate --style "success" --context "lockfile" ":lock: Lockfile is up to date."
	exit 0
fi
