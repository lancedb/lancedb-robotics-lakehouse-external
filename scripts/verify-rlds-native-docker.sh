#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

IMAGE="${LANCEDB_ROBOTICS_RLDS_DOCKER_IMAGE:-lancedb-robotics-rlds-native}"
PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/amd64}"

docker build \
  --platform "$PLATFORM" \
  -f docker/rlds-native.Dockerfile \
  -t "$IMAGE" \
  .

docker run --rm --platform "$PLATFORM" "$IMAGE" "$@"
