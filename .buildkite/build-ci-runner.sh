#!/usr/bin/env bash

set -eo pipefail

GCR_PATH=europe-docker.pkg.dev/embark-shared/main/ml/ci-runner

buildah build-using-dockerfile \
        --tag $GCR_PATH \
        --file Dockerfile \
        --layers \
        --format=docker \
        .

buildah push --digestfile digest.txt --format=v2s2 $GCR_PATH
echo "🚀 CI base image pushed: \`$GCR_PATH@`cat digest.txt`\`"
sed -i "3s|.*|    image: $GCR_PATH@`cat digest.txt`|" pipeline.yml
