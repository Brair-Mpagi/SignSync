#!/usr/bin/env sh
#
# Container entrypoint.
#
# Its whole job is to make a misconfigured container fail at startup with a legible
# message, instead of starting and quietly lacking a feature. A service that accepts
# sign input and recognises nothing looks, from the clinic, exactly like a system
# that does not work — and takes far longer to diagnose than a container that
# refuses to boot.

set -eu

MODEL="${SIGNSYNC_MODEL:-}"
BOOTSTRAP="${SIGNSYNC_BOOTSTRAP_DEMO:-0}"
HOST="${SIGNSYNC_HOST:-0.0.0.0}"
PORT="${SIGNSYNC_PORT:-8000}"

# The demo corpus and model go to ephemeral container storage, not to a mounted
# volume. Two reasons: a bind-mounted host directory is owned by the host user and
# this container runs unprivileged, so writing there fails; and a synthetic demo
# model is scaffolding that should vanish with the container rather than sit in
# artifacts/ where someone could later mistake it for a trained system.
CORPUS="${SIGNSYNC_BOOTSTRAP_CORPUS:-/tmp/demo-corpus}"
DEMO_MODEL="${SIGNSYNC_BOOTSTRAP_MODEL:-/tmp/demo-recogniser.npz}"

echo "SignSync container starting"
echo "---------------------------"

# Optional demo bootstrap: build a synthetic corpus and train on it if no model was
# supplied. Off by default, because a demo model must never be mistaken for a
# trained system — it is synthetic, it contains no USL, and every response it
# produces carries the provisional warnings that say so.
if [ -z "$MODEL" ] && [ "$BOOTSTRAP" = "1" ]; then
  echo "SIGNSYNC_BOOTSTRAP_DEMO=1 and no model supplied — building a synthetic demo model."
  echo "This is scaffolding for a demonstration. It is not trained on Ugandan Sign Language."

  if [ ! -f "$CORPUS/manifest.json" ]; then
    signsync corpus build-synthetic "$CORPUS" --signers 8 --repeats 2
  fi
  signsync train "$CORPUS" --out "$DEMO_MODEL"

  if [ ! -f "$DEMO_MODEL" ]; then
    echo "ERROR: the demo bootstrap did not produce a model at '$DEMO_MODEL'." >&2
    echo "       Training failed; the log above says why." >&2
    exit 1
  fi

  MODEL="$DEMO_MODEL"
  export SIGNSYNC_MODEL="$MODEL"

elif [ -n "$MODEL" ] && [ ! -f "$MODEL" ]; then
  echo "ERROR: SIGNSYNC_MODEL points at '$MODEL', which is not in the container." >&2
  echo "       Mount it, or set SIGNSYNC_BOOTSTRAP_DEMO=1 for a synthetic demo model." >&2
  exit 1
fi

signsync doctor || true
echo

exec uvicorn signsync.api:app --host "$HOST" --port "$PORT" --log-level "${SIGNSYNC_LOG_LEVEL:-info}"
