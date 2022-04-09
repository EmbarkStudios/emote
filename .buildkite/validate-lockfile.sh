set -eo pipefail

EXIT_CODE=0
poetry lock --check || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    buildkite-agent annotate --style "error" --context "lockfile" ":lock: Lockfile is outdated. Please run \`poetry lock --no-update\` and commit the result."
  exit 1
fi
