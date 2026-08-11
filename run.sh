#!/usr/bin/env bash
#
# Run the whole SignSync system, from nothing to a served bidirectional translator.
#
#   ./run.sh                 set up, build a corpus, train, evaluate, serve
#   ./run.sh --help          all options
#
# Why a script and not a README section: the sequence has real dependencies between
# its steps — you cannot train without a corpus, cannot serve a model you have not
# trained, and cannot evaluate without a signer-independent split. Encoding that
# once means a new contributor, a pilot site, and CI all run the same thing.
#
# Everything here works offline (plan §17). No step reaches the network except the
# initial `pip install`, and `--offline` skips even that.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- defaults

VENV=".venv"
CORPUS="data/samples/demo"
MODEL="artifacts/recogniser.npz"
SIGNERS=8
REPEATS=2
HOST="127.0.0.1"
PORT=8000
EXTRAS="dev,api"

DO_SETUP=1
DO_CORPUS=1
DO_TRAIN=1
DO_EVALUATE=1
DO_DEMO=1
DO_SERVE=1
DO_TESTS=0
FORCE=0

# ---------------------------------------------------------------- output

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

step()  { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$RESET"; }
info()  { printf '    %s\n' "$*"; }
warn()  { printf '%s !  %s%s\n' "$YELLOW" "$*" "$RESET" >&2; }
die()   { printf '%s ✗  %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }
ok()    { printf '%s ✓  %s%s\n' "$GREEN" "$*" "$RESET"; }

usage() {
  cat <<EOF
${BOLD}SignSync — run the whole system${RESET}

  ./run.sh [options]

${BOLD}Pipeline stages${RESET} (all run by default, in order)
  --skip-setup       use the current Python environment as-is
  --skip-corpus      reuse an existing corpus
  --skip-train       reuse an existing model
  --skip-evaluate    do not run the evaluation report
  --no-demo          skip the one-exchange demonstration
  --no-serve         stop after the demo instead of starting the server

  --only <stage>     run one stage only: setup|corpus|train|evaluate|serve
  --test             run the test suite before the pipeline

${BOLD}Options${RESET}
  --corpus <dir>     corpus directory            (default: $CORPUS)
  --model <file>     model path                  (default: $MODEL)
  --signers <n>      synthetic signers to make   (default: $SIGNERS)
  --repeats <n>      clips per gloss per signer  (default: $REPEATS)
  --host <addr>      bind address                (default: $HOST)
  --port <n>         port                        (default: $PORT)
  --venv <dir>       virtualenv location         (default: $VENV)
  --extras <list>    pip extras to install       (default: $EXTRAS)
  --force            rebuild the corpus and retrain even if they exist
  --offline          skip pip install entirely (requires signsync importable)
  -h, --help         this message

${BOLD}Examples${RESET}
  ./run.sh                                  # everything, then serve on :$PORT
  ./run.sh --no-serve --test                # what CI does
  ./run.sh --only serve --model my.npz      # serve an existing model
  ./run.sh --skip-setup --force             # rebuild data and model, reuse the venv

For containers, use ${BOLD}infrastructure/run-docker.sh${RESET} instead.
EOF
}

# ---------------------------------------------------------------- arguments

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)    DO_SETUP=0 ;;
    --skip-corpus)   DO_CORPUS=0 ;;
    --skip-train)    DO_TRAIN=0 ;;
    --skip-evaluate) DO_EVALUATE=0 ;;
    --no-demo)       DO_DEMO=0 ;;
    --no-serve)      DO_SERVE=0 ;;
    --test)          DO_TESTS=1 ;;
    --force)         FORCE=1 ;;
    --offline)       DO_SETUP=0 ;;
    --only)
      [[ $# -ge 2 ]] || die "--only needs a stage"
      DO_SETUP=0; DO_CORPUS=0; DO_TRAIN=0; DO_EVALUATE=0; DO_SERVE=0; DO_DEMO=0
      case "$2" in
        setup)    DO_SETUP=1 ;;
        corpus)   DO_CORPUS=1 ;;
        train)    DO_TRAIN=1 ;;
        evaluate) DO_EVALUATE=1 ;;
        serve)    DO_SERVE=1 ;;
        *) die "unknown stage '$2' (setup|corpus|train|evaluate|serve)" ;;
      esac
      shift ;;
    --corpus)  CORPUS="${2:?--corpus needs a path}"; shift ;;
    --model)   MODEL="${2:?--model needs a path}"; shift ;;
    --signers) SIGNERS="${2:?--signers needs a number}"; shift ;;
    --repeats) REPEATS="${2:?--repeats needs a number}"; shift ;;
    --host)    HOST="${2:?--host needs an address}"; shift ;;
    --port)    PORT="${2:?--port needs a number}"; shift ;;
    --venv)    VENV="${2:?--venv needs a path}"; shift ;;
    --extras)  EXTRAS="${2:?--extras needs a list}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

# ---------------------------------------------------------------- environment

PY="python3"

activate_venv() {
  if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PY="python"
  fi
}

if [[ $DO_SETUP -eq 1 ]]; then
  step "Setting up the Python environment"
  command -v python3 >/dev/null || die "python3 not found"

  if [[ ! -d "$VENV" ]]; then
    info "creating $VENV"
    python3 -m venv "$VENV"
  else
    info "reusing $VENV"
  fi
  activate_venv

  info "installing signsync[$EXTRAS]"
  # Quiet unless it fails: a wall of pip output buries the steps that matter.
  if ! python -m pip install --quiet --upgrade pip 2>/dev/null; then
    warn "could not upgrade pip; continuing"
  fi
  python -m pip install --quiet -e ".[$EXTRAS]" \
    || die "pip install failed. Offline? Try: ./run.sh --offline --skip-setup"
  ok "environment ready"
else
  activate_venv
fi

# The package must be importable from here on, whichever way we got there.
if ! $PY -c "import signsync" 2>/dev/null; then
  if [[ -d src/signsync ]]; then
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    info "using src/ layout via PYTHONPATH"
  fi
fi
$PY -c "import signsync" 2>/dev/null \
  || die "signsync is not importable. Run without --skip-setup, or: pip install -e '.[$EXTRAS]'"

SIGNSYNC="$PY -m signsync.cli"

step "Capabilities"
$SIGNSYNC doctor

if [[ $DO_TESTS -eq 1 ]]; then
  step "Test suite"
  $PY -m pytest -q || die "tests failed"
  ok "tests passed"
fi

# ---------------------------------------------------------------- corpus

if [[ $DO_CORPUS -eq 1 ]]; then
  step "Corpus"
  if [[ -f "$CORPUS/manifest.json" && $FORCE -eq 0 ]]; then
    info "reusing $CORPUS (--force to rebuild)"
  else
    [[ $FORCE -eq 1 && -d "$CORPUS" ]] && rm -rf "$CORPUS"
    info "generating a synthetic corpus in $CORPUS"
    # Synthetic: derived from no real person, and it contains no USL. It exists so
    # the system is runnable before any signer has been recorded (plan §9.1).
    $SIGNSYNC corpus build-synthetic "$CORPUS" --signers "$SIGNERS" --repeats "$REPEATS"
  fi
  $SIGNSYNC corpus stats "$CORPUS"
fi

# ---------------------------------------------------------------- train

if [[ $DO_TRAIN -eq 1 ]]; then
  step "Training the recogniser"
  if [[ -f "$MODEL" && $FORCE -eq 0 ]]; then
    info "reusing $MODEL (--force to retrain)"
  else
    [[ -f "$CORPUS/manifest.json" ]] || die "no corpus at $CORPUS; run without --skip-corpus"
    mkdir -p "$(dirname "$MODEL")"
    $SIGNSYNC train "$CORPUS" --out "$MODEL"
  fi
fi

# ---------------------------------------------------------------- evaluate

if [[ $DO_EVALUATE -eq 1 ]]; then
  step "Evaluation"
  if [[ ! -f "$MODEL" ]]; then
    warn "no model at $MODEL; skipping evaluation"
  else
    # This exits non-zero until a certified human evaluation round exists, which is
    # the designed behaviour (plan §15) and not a failure of this script.
    if $SIGNSYNC evaluate "$CORPUS" "$MODEL"; then
      ok "evaluation supports a success claim"
    else
      warn "evaluation does not support a success claim — see the blockers above."
      warn "That is expected without human evaluation. Plan §15 makes it mandatory."
    fi
  fi
fi

# ---------------------------------------------------------------- demo + serve

if [[ $DO_DEMO -eq 1 ]]; then
  step "One exchange in each direction"
  $SIGNSYNC demo speech-to-sign --text "Where is the hospital?"
  echo
  if [[ -f "$MODEL" ]]; then
    $SIGNSYNC demo sign-to-speech --synthetic --glosses ME NEED HELP --model "$MODEL"
  else
    $SIGNSYNC demo sign-to-speech --glosses ME NEED HELP
  fi
fi

if [[ $DO_SERVE -eq 1 ]]; then
  step "Serving"
  info "browser client : http://$HOST:$PORT"
  info "API docs       : http://$HOST:$PORT/docs"
  info "health         : http://$HOST:$PORT/health"
  printf '\n%sOutput is provisional. Not a substitute for a qualified interpreter.%s\n\n' \
    "$DIM" "$RESET"

  serve_args=(serve --host "$HOST" --port "$PORT")
  [[ -f "$MODEL" ]] && serve_args+=(--model "$MODEL")
  exec $SIGNSYNC "${serve_args[@]}"
fi

ok "done"
