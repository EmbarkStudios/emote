set -eo pipefail

EXIT_CODE=0
pdm lock --refresh || EXIT_CODE=$?

GIT_STATUS=$(git status --porcelain --untracked-files=no -- pdm.lock)
if [ -n "$GIT_STATUS" ]; then
    buildkite-agent annotate --style "error" --context "lockfile" ":lock: Lockfile is outdated. Please run \`pdm lock --no-update\` and commit the result."
	exit 1
else
    buildkite-agent annotate --style "success" --context "lockfile" ":lock: Lockfile is up to date."
fi
