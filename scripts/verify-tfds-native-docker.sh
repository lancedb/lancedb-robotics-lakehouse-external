#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

IMAGE="${LANCEDB_ROBOTICS_TFDS_DOCKER_IMAGE:-lancedb-robotics-tfds-native}"
if [ -n "${DOCKER_DEFAULT_PLATFORM:-}" ]; then
  PLATFORM="$DOCKER_DEFAULT_PLATFORM"
else
  case "$(docker info --format '{{.Architecture}}')" in
    amd64|x86_64) PLATFORM="linux/amd64" ;;
    arm64|aarch64) PLATFORM="linux/arm64" ;;
    *)
      echo "cannot map Docker engine architecture; set DOCKER_DEFAULT_PLATFORM" >&2
      exit 2
      ;;
  esac
fi

docker build \
  --platform "$PLATFORM" \
  -f docker/tfds-native.Dockerfile \
  -t "$IMAGE" \
  .

docker run --rm --platform "$PLATFORM" "$IMAGE" "$@"
