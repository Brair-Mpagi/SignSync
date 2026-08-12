# SignSync

[![CI](https://github.com/Brair-Mpagi/SignSync/actions/workflows/ci.yml/badge.svg)](https://github.com/Brair-Mpagi/SignSync/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](pyproject.toml)

Real-time, two-way translation between **Ugandan Sign Language** and spoken English.

Point a camera at someone signing and get spoken English out. Speak, and a 3D avatar signs back.
Runs on a laptop, works offline, no GPU required.

```
      🤟  signing  ──▶  landmarks ──▶ recognition ──▶ ┐
                                                      ├──▶  meaning  ──┐
      🎙️  speech   ──▶  transcript ──▶ parsing  ──────▶ ┘                │
                                                                       │
                       ┌───────────────────────────────────────────────┘
                       │
                       ├──▶  English sentence  ──▶  🔊  speech
                       └──▶  USL gloss + markers ──▶ 🤟  3D avatar
```

> **Heads up:** the sign vocabulary and grammar rules that ship here are working placeholders, not
> reviewed Ugandan Sign Language. See [Project status](#project-status) before showing output to
> anyone.

---

## Quick start

```bash
git clone https://github.com/Brair-Mpagi/SignSync.git
cd SignSync
./run.sh
```

That sets up a virtualenv, generates a practice corpus, trains a recogniser, evaluates it, and
serves the app at **http://localhost:8000**. Takes about a minute; needs nothing but Python 3.10+.

Prefer Docker?

```bash
./infrastructure/run-docker.sh --demo
```

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
```

The core install is deliberately tiny — NumPy and nothing else. Everything heavier is optional, and
the app degrades gracefully without it:

| Extra | Adds | Enables |
|---|---|---|
| `vision` | MediaPipe, OpenCV | Live camera tracking |
| `models` | PyTorch | Trainable LSTM / TCN / Transformer recognisers |
| `speech` | Whisper, Piper | Microphone input and spoken output |
| `api` | FastAPI, Uvicorn | Web server and browser client |
| `runtime` | ONNX Runtime | Faster CPU inference for deployment |

```bash
pip install -e ".[vision,models,speech]"
```

Not sure what your machine has? `signsync doctor` lists every capability, what it enables, and what
happens without it.

---

## Usage

### Command line

```bash
# Translate, either direction
signsync translate english-to-sign "Where is the hospital?"
signsync translate sign-to-english ME NEED HELP

# Add --trace to see the intermediate meaning representation
signsync translate english-to-sign "I do not understand" --trace

# Run a full exchange through the pipeline
signsync demo speech-to-sign --text "I need help"
signsync demo sign-to-speech --synthetic --model artifacts/recogniser.npz

# Serve the API and browser client
signsync serve --model artifacts/recogniser.npz
```

### Training your own recogniser

```bash
signsync corpus build-synthetic data/samples/demo --signers 8
signsync corpus stats  data/samples/demo          # consent + diversity report
signsync train         data/samples/demo --out artifacts/recogniser.npz
signsync evaluate      data/samples/demo artifacts/recogniser.npz
```

Splits are always by signer, never by clip — so the accuracy you see is accuracy on people the
model has never met.

### HTTP API

Start the server, then:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Status, capabilities, active warnings |
| `POST` | `/api/sign-to-english` | Glosses → English (+ optional speech) |
| `POST` | `/api/english-to-sign` | English → glosses + avatar animation |
| `POST` | `/api/speech-to-sign` | Transcript → avatar animation |
| `GET` | `/api/rig` | Avatar skeleton definition |
| `GET` | `/api/lexicon` | Vocabulary and validation state |
| `GET` | `/api/metrics` | Per-stage latency |
| `WS` | `/ws/sign` | Stream landmarks in, translations out |
| `WS` | `/ws/speak` | Stream text in, animations out |

```bash
curl -X POST http://localhost:8000/api/english-to-sign \
  -H 'Content-Type: application/json' \
  -d '{"text": "Where is the hospital?"}'
```

```json
{
  "glosses": ["HOSPITAL", "WHERE"],
  "notation": "______________wh\nHOSPITAL WHERE",
  "markers": [{ "marker": "brow_furrow", "scope": ["HOSPITAL", "WHERE"] }],
  "animation": { "fps": 30.0, "frames": ["… 91 frames of joint rotations …"] },
  "generated": ["HOSPITAL", "WHERE"],
  "missing": [],
  "warnings": [
    { "code": "unvalidated_lexicon", "message": "The sign lexicon has not been reviewed…" },
    { "code": "generated_motion", "message": "Some signs were rendered from generated motion…" }
  ]
}
```

The question word moves to the end and the brow-furrow marker scopes the whole clause — that's USL
grammar, not a formatting quirk. Note the `warnings`: every response tells you what to distrust
about it. Interactive docs at `/docs`.

### Docker

```bash
./infrastructure/run-docker.sh --demo                       # build + run, trains a demo model
./infrastructure/run-docker.sh --model artifacts/model.npz  # run with your own model
./infrastructure/run-docker.sh --detach                     # background
docker compose -f infrastructure/docker-compose.yml up      # if you use compose
```

Configure with environment variables — `SIGNSYNC_MODEL`, `SIGNSYNC_LEXICON`, `SIGNSYNC_CLIPS`,
`SIGNSYNC_VOICE`, `SIGNSYNC_MIN_CONFIDENCE`, `SIGNSYNC_REQUIRE_MODEL`. Full table in
[infrastructure/README.md](infrastructure/README.md).

---

## How it works

Sign language is not English on the hands. Word order, negation and question marking all work
differently, so translating sign-to-word produces nonsense in both directions. Everything here
routes through a **semantic frame** instead — who did what to whom, what kind of utterance it is,
whether it is negated — and each language is generated from that.

```
glosses ──▶ parse ──▶ ┌──────────────┐ ──▶ realise ──▶ English
                      │ SemanticFrame│
English ──▶ parse ──▶ └──────────────┘ ──▶ generate ──▶ glosses + markers
```

A few consequences worth knowing:

- **Landmarks, not pixels.** Video becomes ~40 tracked points per frame, so models are small, run on
  CPU, and generalise across skin tone, lighting and clothing.
- **Facial expression is grammar.** Brow position marks questions, head shake marks negation. These
  travel through the whole pipeline as first-class data and drive the avatar's face.
- **Left-handed signers are handled.** Signing space is canonicalised to dominant/non-dominant, so a
  left-handed signer isn't a whole second vocabulary to learn.
- **Nothing is silently faked.** No sign for a word? Reported. No motion clip? Reported. Recogniser
  unsure? It abstains instead of guessing. Every API response carries a `warnings` array, and the
  clients display it.

---

## Project structure

```
src/signsync/
├── vision/         camera → landmarks → normalised features
├── recognition/    temporal models, training, continuous segmentation
├── translation/    semantic frame, both directions
├── speech/         speech-to-text and text-to-speech adapters
├── motion/         gloss → avatar motion, blending, inverse kinematics
├── avatar/         skeleton rig and animation format
├── datasets/       corpus schema, consent, signer-independent splits
├── evaluation/     metrics and human-evaluation tooling
├── api/            FastAPI server + zero-dependency browser client
├── pipeline.py     ties it all together
└── config.py       environment configuration

frontend/           React + TypeScript + Three.js client
infrastructure/     Dockerfile, compose, deployment notes
docs/               plan, limitations, data protection
```

---

## Development

```bash
make dev        # install with dev tooling
make test       # pytest  (397 tests)
make lint       # ruff
make typecheck  # mypy
make check      # all three, same as CI
```

The test suite runs with only NumPy installed — CI actively asserts that PyTorch, MediaPipe and
OpenCV are *absent* in the core job, so the offline path can't quietly rot.

---

## Project status

Working end-to-end, on synthetic data. Every stage is implemented and wired together, and the whole
thing runs offline. What it needs before it can translate real USL:

- [ ] **A USL video corpus.** No public one exists. It has to be recorded with Deaf signers, under
      consent, with UNAD and Kyambogo University.
- [ ] **A reviewed lexicon and grammar.** The shipped `usl_lexicon.json` is a placeholder built to
      exercise the pipeline. `Lexicon.is_validated` stays `False` until a linguist signs it off.
- [ ] **Trained weights.** Waiting on the corpus.
- [ ] **Human evaluation.** `signsync evaluate` reports 100% accuracy on synthetic data and *still*
      refuses to call that success — automatic metrics can't tell you whether a translation means
      the right thing. Only Deaf evaluators can.

Full detail in [docs/limitations.md](docs/limitations.md). The project plan this implements is
[docs/plan.md](docs/plan.md).

**This is an assistive tool, not an interpreter.** Don't rely on it in medical, legal, or
safety-critical situations.

---

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

One rule that isn't negotiable by pull request: **changes to vocabulary, grammar, or avatar motion
need review by a fluent signer.** An engineer can't self-approve how a sign looks.

Recorded participant data never goes in the repo. `data/` is git-ignored; read
[docs/data-protection.md](docs/data-protection.md) first.

---

## License

MIT, as declared in [pyproject.toml](pyproject.toml). A `LICENSE` file has not been added to the
repository yet.

Corpus data is *not* covered by it. Recordings are governed by the consent agreements signed with
each participant.
