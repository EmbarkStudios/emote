set -eo pipefail

pdm plugin add plugins/pdm-plugin-torch

EXIT_CODE=0
pdm lock --check || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    buildkite-agent annotate --style "error" --context "lockfile" ":lock: Failed validating lockfile. See logs for more info."
	exit 1
fi

TORCH_EXIT_CODE=0
pdm torch lock --check || TORCH_EXIT_CODE=$?
GIT_STATUS=$(git status --porcelain --untracked-files=no -- pdm.lock)
if [ -n "$GIT_STATUS" ] && [ $TORCH_EXIT_CODE -ne 0 ]; then
	lock_diff=$(git diff pdm.lock)
	cat << EOF | buildkite-agent annotate --style "error" --context "lockfile"
:lock: Lockfiles are outdated. Please run \`pdm lock --no-update && pdm torch lock\` and commit the result.

\`\`\`diff
$lock_diff
\`\`\`
EOF
	exit 1
elif  [ -n "$GIT_STATUS" ] ; then
	lock_diff=$(git diff pdm.lock)
	cat << EOF | buildkite-agent annotate --style "error" --context "lockfile"
:lock: Lockfile is outdated. Please run \`pdm lock --no-update\` and commit the result.

\`\`\`diff
$lock_diff
\`\`\`
EOF
	exit 1
elif [ $TORCH_EXIT_CODE -ne 0 ]; then

	cat << EOF | buildkite-agent annotate --style "error" --context "lockfile"
:lock: Torch lockfile is outdated. Please run \`pdm torch lock\` and commit the result.
EOF
else
    buildkite-agent annotate --style "success" --context "lockfile" ":lock: Lockfile is up to date."
fi
