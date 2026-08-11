# frontend/

React + TypeScript + Three.js client — the stack named in [plan §10](../docs/plan.md).

## Status

**Source only. Not built, and not verified against a running build.** The environment
this was written in had no package registry access, so `npm install` never ran and neither did
`tsc`. Treat this as a reviewed starting point, not as working software: expect to fix type
errors and version drift on first install.

The client that *is* verified is the dependency-free one at
[src/signsync/api/static/](../src/signsync/api/static/). It is served by `signsync serve`, renders
the same animation format on a 2D canvas, and its forward kinematics is cross-checked against the
Python implementation in [tests/test_client.py](../tests/test_client.py). Plan §17 expects
deployments with no connectivity and no build toolchain, so that client — not this one — is the
one that ships with the server.

## Why both

| | `src/signsync/api/static/` | `frontend/` |
|---|---|---|
| Dependencies | none | React, Three.js, Vite |
| Build step | none | `npm run build` |
| Rendering | 2D canvas skeleton | 3D rigged avatar (WebGL) |
| Ships with the server | yes | no — deploy separately |
| Works offline on first run | yes | after a build |

Both speak the same API, so a deployment can choose based on what its hardware and network can
support (plan §17, "low hardware bar").

## Run

```bash
npm install
npm run dev          # expects the API on http://localhost:8000
npm run build
```

## Layout

```
src/
├── api.ts                    API client and response types
├── kinematics.ts             forward kinematics — must match signsync/avatar/rig.py
├── App.tsx                   the three modes of plan §18.3
└── components/
    ├── Avatar.tsx            Three.js rigged avatar
    ├── SignInput.tsx         Mode A: camera/gloss input
    └── Warnings.tsx          confidence and warning display
```

## The rule this client must not break

Plan §16.3 requires the product to be transparent about its limits. Every response carries
`warnings`, `confidence`, `missing` and `generated`. **Render all of them.** A build of this client
that hides the confidence indicator or drops the "not a certified interpreter" disclaimer is not a
styling choice — it is the failure mode plan §14 calls "community trust", and it should not pass
review.
