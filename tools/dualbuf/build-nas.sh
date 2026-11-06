#!/usr/bin/env bash
# Build the NAS (ROCm/HIP, gfx1030 target) image from the dual-buffer branch of the fork.
# Usage: tools/dualbuf/build-nas.sh [IMAGE_TAG]    (default: llama-dualbuf:rocm)
# Env:   BRANCH (dual-buffer), REPO_URL, ROCM_TAG (7.0-complete), GPU_TARGET (gfx1030),
#        GH_TOKEN (only if the fork is private; passed as a BuildKit secret, not stored in the image)
# The branch must be pushed to the fork first; the image clones it fresh on every build.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-llama-dualbuf:rocm}"
args=(--build-arg "CACHE_BUST=$(date +%s)")
for v in BRANCH REPO_URL ROCM_TAG GPU_TARGET; do
  [ -n "${!v:-}" ] && args+=(--build-arg "$v=${!v}")
done
[ -n "${GH_TOKEN:-}" ] && args+=(--secret id=gh_token,env=GH_TOKEN)
DOCKER_BUILDKIT=1 docker build "${args[@]}" -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
cat <<MSG
built: $TAG
run:   docker run --rm -it --device /dev/kfd --device /dev/dri/renderD129 --group-add video \
         -v /mnt/nvme/ollama/models:/models:ro -v /mnt/nvme/dualbuf:/work $TAG
MSG
