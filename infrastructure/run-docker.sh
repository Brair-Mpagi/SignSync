#!/usr/bin/env bash
#
# Build and run SignSync in a container, using plain `docker` only.
#
#   ./infrastructure/run-docker.sh              build and serve on :8000
#   ./infrastructure/run-docker.sh --help       all options
#
# Compose does the same thing with more configuration, but the compose plugin is a
# separate install that plenty of machines do not have — including some of the pilot
# hardware in plan §17. This script needs nothing but the docker CLI, so
# "run it in a container" never depends on a second tool being present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="signsync:latest"
NAME="signsync"
PORT=8000
EXTRAS="api"
MODEL=""
BOOTSTRAP=0
DO_BUILD=1
DETACH=0
SHELL_MODE=0
WRITABLE_ARTIFACTS=0

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; BLUE=""; RESET=""
fi
step() { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$RESET"; }
die()  { printf '%s ✗  %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }
ok()   { printf '%s ✓  %s%s\n' "$GREEN" "$*" "$RESET"; }

usage() {
  cat <<EOF
${BOLD}Run SignSync in a container${RESET}

  ./infrastructure/run-docker.sh [options]

  --port <n>       host port                        (default: $PORT)
  --name <name>    container name                   (default: $NAME)
  --image <ref>    image tag to build/run           (default: $IMAGE)
  --extras <list>  pip extras baked into the image  (default: $EXTRAS)
  --model <file>   host path to a trained model, mounted read-only
  --demo           train a synthetic demo model on first boot (kept in the container)
  --writable-artifacts
                   mount artifacts/ writable and run as your uid, so the container
                   can save models to the host
  --skip-build     run an existing image
  --detach         run in the background
  --shell          open a shell in the image instead of serving
  -h, --help       this message

${BOLD}Examples${RESET}
  ./infrastructure/run-docker.sh --demo
  ./infrastructure/run-docker.sh --model artifacts/recogniser.npz
  ./infrastructure/run-docker.sh --extras api,models --skip-build
  ./infrastructure/run-docker.sh --writable-artifacts   # container can save models
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)       PORT="${2:?--port needs a number}"; shift ;;
    --name)       NAME="${2:?--name needs a value}"; shift ;;
    --image)      IMAGE="${2:?--image needs a value}"; shift ;;
    --extras)     EXTRAS="${2:?--extras needs a list}"; shift ;;
    --model)      MODEL="${2:?--model needs a path}"; shift ;;
    --demo)       BOOTSTRAP=1 ;;
    --writable-artifacts) WRITABLE_ARTIFACTS=1 ;;
    --skip-build) DO_BUILD=0 ;;
    --detach)     DETACH=1 ;;
    --shell)      SHELL_MODE=1 ;;
    -h|--help)    usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable"

if [[ $DO_BUILD -eq 1 ]]; then
  step "Building $IMAGE (extras: $EXTRAS)"
  docker build \
    --file infrastructure/Dockerfile \
    --build-arg "EXTRAS=$EXTRAS" \
    --tag "$IMAGE" \
    . || die "build failed"
  ok "built $IMAGE"
fi

# A previous run of the same name would make `docker run` fail with a name clash,
# which is a confusing way to learn that the last container is still up.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  step "Removing the previous '$NAME' container"
  docker rm -f "$NAME" >/dev/null
fi

mkdir -p data artifacts

run_args=(
  --name "$NAME"
  --publish "${PORT}:8000"
  # Corpus data read-only: an image or a compromised service has no business
  # writing to participant recordings (docs/data-protection.md).
  --volume "$REPO_ROOT/data:/data:ro"
  --security-opt no-new-privileges:true
  --read-only
  --tmpfs /tmp
)

# Which host directory becomes /artifacts. When --model is given it is the model's
# own directory, so the model arrives as an ordinary file inside a single mount.
# Mounting the file *into* an already-mounted /artifacts cannot work: docker has to
# create the mountpoint first, and the directory mount is read-only by then.
artifacts_src="$REPO_ROOT/artifacts"
if [[ -n "$MODEL" ]]; then
  [[ -f "$MODEL" ]] || die "no model at '$MODEL'"
  artifacts_src="$(cd "$(dirname "$MODEL")" && pwd)"
  run_args+=(--env "SIGNSYNC_MODEL=/artifacts/$(basename "$MODEL")")
elif [[ $BOOTSTRAP -eq 1 ]]; then
  run_args+=(--env "SIGNSYNC_BOOTSTRAP_DEMO=1")
fi

# artifacts/ is read-only by default: serving reads a model and never writes one.
# Making it writable also means running as the host uid, because a bind mount keeps
# the host's ownership and the image's unprivileged user cannot write to it.
if [[ $WRITABLE_ARTIFACTS -eq 1 ]]; then
  run_args+=(--volume "$artifacts_src:/artifacts" --user "$(id -u):$(id -g)")
else
  run_args+=(--volume "$artifacts_src:/artifacts:ro")
fi

if [[ $SHELL_MODE -eq 1 ]]; then
  step "Shell in $IMAGE"
  exec docker run --rm -it --entrypoint /bin/bash "${run_args[@]}" "$IMAGE"
fi

step "Starting $NAME on http://localhost:$PORT"
if [[ $DETACH -eq 1 ]]; then
  docker run --detach "${run_args[@]}" "$IMAGE" >/dev/null

  printf '    waiting for /health'
  for _ in $(seq 1 40); do
    if docker exec "$NAME" python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2)" 2>/dev/null; then
      printf '\n'; ok "healthy: http://localhost:$PORT"
      echo "    logs: docker logs -f $NAME"
      echo "    stop: docker rm -f $NAME"
      exit 0
    fi
    printf '.'; sleep 1
  done
  printf '\n'
  docker logs --tail 40 "$NAME" >&2
  die "container did not become healthy"
fi

exec docker run --rm -it "${run_args[@]}" "$IMAGE"
