# Contributing

## Before you start

Read [docs/plan.md](docs/plan.md) §6 and §16, and [docs/limitations.md](docs/limitations.md). This
project has process rules that are not negotiable by a pull request.

## Review rules

| Change touches | Required reviewer |
|---|---|
| Lexicon entries, gloss schema, grammar rules | A fluent USL signer or a Kyambogo/UNASLI linguist |
| Avatar motion, non-manual markers, timing | A Deaf evaluator — plan §14, "robotic avatar" mitigation |
| Consent handling, retention, data export | Project lead + whoever holds data-protection responsibility |
| Anything else | Ordinary code review |

An engineer cannot self-approve a change to how a sign looks. That is the rule that separates this
project from "a gesture classifier called a translator".

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check          # ruff + mypy + pytest
```

The default test run must pass with **only** the core dependency (`numpy`) installed. Tests needing
an optional extra must be marked and skipped cleanly:

```python
torch = pytest.importorskip("torch")          # or
@pytest.mark.requires_torch
```

If a change makes the suite fail without `torch`, `mediapipe`, or a network connection, that is a
bug in the change, not in the environment — plan §17 requires the system to work offline on modest
hardware.

## Commit style

Conventional commits scoped by pipeline stage:

```
feat(vision): normalise landmarks against shoulder width
fix(datasets): reject splits where a signer appears in both sides
docs(plan): record Phase 2 evaluation outcome
```

## Never commit

Recorded video, audio, landmark files, or consent records. See
[docs/data-protection.md](docs/data-protection.md). If you have already committed one, say so
immediately — the history has to be rewritten, and that is much easier the same day.
