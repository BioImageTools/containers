#!/bin/bash

set -euo pipefail

readonly path="$1"
readonly container=$(echo $path | cut -d '/' -f 1)
readonly version=$(echo $path | cut -d '/' -f 2)

now="$(date --utc --iso-8601=seconds)"
repo_url="https://github.com/BioImageTools/containers"
tag="ghcr.io/bioimagetools/$container:$version"

docker build \
    --label org.opencontainers.image.source="$repo_url" \
    --label org.opencontainers.image.created="$now" \
    --tag $tag \
    ./$container/$version

read -r -p "Do you want to push this image to GCR? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY])
        docker push $tag
        ;;
    *)
        echo "You can push it later like this: docker push $tag"
        ;;
esac

