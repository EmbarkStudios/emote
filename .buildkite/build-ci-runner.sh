#!/usr/bin/env bash

set -eo pipefail

GCR_PATH=gcr.io/embark-shared/ml/ci-runner

buildah build-using-dockerfile \
        --tag $GCR_PATH \
        --file Dockerfile \
        --layers \
        --format=docker \
		--squash \
        .

buildah push --digestfile digest.txt --format=v2s2 $GCR_PATH
echo "🚀 CI base image pushed: \`$GCR_PATH@`cat digest.txt`\`"
