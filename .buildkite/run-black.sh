set -eo pipefail

echo --- Installing dependencies

echo $(which buildkite-agent)
exit 0

apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential

export PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.2.0b1 \
    POETRY_HOME="/tmp/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \

pip install poetry==1.2.0b1
poetry install
poetry env info --path

echo --- Running black
EXIT_CODE=0
poetry run black --diff emote > diff.txt || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
	cat << EOF | buildkite-agent annotate --style "error" --context "eslint"
:error: Your code isn't formatted by `black`. Please fix the below diffs, or run `poetry run black` to automatically format it.

\`\`\`diff
$(cat diff.txt)
\`\`\`

EOF
else
	buildkite-agent annotate ":check: Code formatted correctly " --style "success" --context "eslint"
fi
