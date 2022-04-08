crlf_count=$(find . -not -type d -exec file "{}" ";" | grep CRLF | wc -l)

if [[ $crlf_count -gt 0 ]]; then
    buildkite-agent annotate --style "error" --context validate-changes "Repository contains CRLF line-endings. To avoid diff issues and cross-platform issues we require that all commits are done using a LF-style. 

If you're doing development on Windows, use \`git config --global core.autocrlf true\` to let Git fix this for you on commit."
    exit 1
fi
