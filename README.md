# SignSync

**Bidirectional Ugandan Sign Language (USL) ↔ spoken English translation.**

SignSync is the software implementation of the project plan in [docs/plan.md](docs/plan.md). It is
built around one principle taken from that plan:

> Do not build a gesture classifier and call it a translator. Build a language system that happens
> to use vision as one of its inputs and motion as one of its outputs.

Everything here therefore routes through an explicit **semantic intermediate representation**
(`signsync.translation.semantics`) rather than mapping sign → English word.

---

## Status

This repository contains the **engineering scaffold and reference implementation** of the pipeline:
every stage exists, is wired together, and runs end-to-end offline on synthetic/sample data.

What it does **not** contain, and cannot contain until the community and data work in the plan
happens:

| Missing | Why | Plan reference |
|---|---|---|
| A real USL corpus | No public continuous USL video corpus exists; it must be collected with UNAD/Kyambogo under consent | §9.1, §16 |
| Trained model weights | Depend on that corpus | §9.2 |
| A validated USL lexicon/grammar | `src/signsync/resources/usl_lexicon.json` is a **placeholder seed** and must be replaced by linguist-reviewed entries | §6, §11 |
| Human evaluation results | Requires the Deaf Advisory Board | §15 |

The lexicon and grammar rules shipped here are structurally correct but linguistically provisional.
**Do not present output from this repository as validated USL** — see
[docs/limitations.md](docs/limitations.md).

---

## Layout

The plan's Appendix C structure, mapped onto a `src/` Python package:

| plan §22 | here |
|---|---|
| `vision/` | [src/signsync/vision/](src/signsync/vision/) — tracking, landmarks, normalisation, features |
| `recognition/` | [src/signsync/recognition/](src/signsync/recognition/) — temporal models, training, inference, continuous segmentation |
| `translation/` | [src/signsync/translation/](src/signsync/translation/) — semantic IR, `sign_to_english/`, `english_to_sign/` |
| `speech/` | [src/signsync/speech/](src/signsync/speech/) — `stt/`, `tts/` adapters |
| `motion/` | [src/signsync/motion/](src/signsync/motion/) — clip library, blending, inverse kinematics |
| `avatar/` | [src/signsync/avatar/](src/signsync/avatar/) — rig definition, animation export |
| `datasets/` | [src/signsync/datasets/](src/signsync/datasets/) — corpus schema, consent registry, splits, augmentation |
| `evaluation/` | [src/signsync/evaluation/](src/signsync/evaluation/) — automatic metrics, human-evaluation tooling |
| `api/` | [src/signsync/api/](src/signsync/api/) — FastAPI backend + realtime WebSocket pipeline |
| `frontend/` | [frontend/](frontend/) — React + TypeScript + Three.js client |
| `infrastructure/` | [infrastructure/](infrastructure/) — Docker, compose, CI |

Recorded data lives in `data/` and is **git-ignored by default**. See
[docs/data-protection.md](docs/data-protection.md) before putting anything there.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core: numpy only
```

Optional extras, each independently installable — the core degrades gracefully without them:

```bash
pip install -e ".[vision]"    # mediapipe + opencv: live camera tracking
pip install -e ".[models]"    # torch: trainable LSTM/TCN/Transformer recognisers
pip install -e ".[speech]"    # whisper-class STT + piper TTS
pip install -e ".[api]"       # fastapi + uvicorn: backend service
pip install -e ".[runtime]"   # onnxruntime: optimised deployment inference
```

`signsync doctor` reports which capabilities are available in the current environment and what each
missing one disables.

## Run

```bash
signsync doctor                                  # capability report
signsync demo sign-to-speech --synthetic         # Mode A, no camera needed
signsync demo speech-to-sign --text "where is the hospital"   # Mode B
signsync serve                                   # FastAPI + browser client on :8000
pytest                                           # test suite (skips what isn't installed)
```

## Design notes

- **Landmarks, not pixels.** Frames become normalised `(x, y, z)` landmark vectors
  (`vision.normalise`) so models train on a modest corpus, run on CPU, and generalise across skin
  tone, lighting and clothing — plan §8.1.
- **Signer-independent by construction.** `datasets.splits` refuses to produce a split where a
  signer appears on both sides; the failure mode it prevents (§14, Risk 2) is silent and fatal.
- **Consent is enforced in code, not policy.** `datasets.consent` gates every clip; withdrawn or
  expired consent removes clips from loading, not just from a spreadsheet — plan §16.
- **Non-manual markers are grammar.** They ride through the whole pipeline as first-class fields on
  the semantic frame and as animation channels on the rig, because dropping them changes sentence
  meaning — plan §8.7.
- **Optional heavy dependencies.** Every third-party model runtime sits behind an adapter with a
  working offline fallback, so the pipeline is demonstrable in a clinic with no internet — plan §17.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes affecting sign quality, lexicon entries, grammar
rules or avatar motion require review by a fluent signer — that is a process rule, not a suggestion.

## Licence

MIT ([LICENSE](LICENSE)). Corpus data is **not** covered by this licence; its terms are set
by the consent agreements with participating signers and by the custodianship arrangement in plan
§17.
