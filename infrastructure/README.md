# infrastructure/

Deployment for the local-first strategy in [plan §17](../docs/plan.md).

## The shape this assumes

Plan §17 pilots in "a health clinic, a school for the Deaf, a government service desk". The
realistic deployment there is **one machine on a local network**, often without reliable internet,
sometimes without any. So:

- One container. The API also serves the browser client, so a pilot site runs a single service.
- No GPU, no CUDA base image. Landmark-based models (plan §8.1) are chosen so CPU inference is
  enough; a multi-gigabyte GPU image would contradict the reason for that choice.
- Optional extras are build arguments, and the default is the smallest set that serves both
  directions: `api` only. `vision`, `models` and `runtime` each add hundreds of megabytes, and none
  is needed when browser clients send landmarks and the NumPy recogniser serves them. Every extra
  is opt-in, because this image reaches sites over slow links and sometimes on a USB stick.

## Run

`run-docker.sh` needs nothing but the docker CLI. The compose plugin is a separate install that
plenty of machines lack — including some of the pilot hardware plan §17 describes — so "run it in a
container" does not depend on a second tool being present.

```bash
./infrastructure/run-docker.sh --demo               # build, train a synthetic model, serve
./infrastructure/run-docker.sh --model artifacts/recogniser.npz
./infrastructure/run-docker.sh --detach             # background, waits for /health
./infrastructure/run-docker.sh --shell              # poke around inside the image
./infrastructure/run-docker.sh --help
```

With compose, if you have it:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
SIGNSYNC_BOOTSTRAP_DEMO=1 docker compose -f infrastructure/docker-compose.yml up
docker compose -f infrastructure/docker-compose.yml --profile frontend up  # + React client
```

Baking in the trainable models instead of the default lightweight set:

```bash
./infrastructure/run-docker.sh --extras api,models,runtime
```

## Configuration

The container's only configuration channel is the environment, read in one place
(`signsync/config.py`) so the mapping is greppable.

| Variable | Effect |
|---|---|
| `SIGNSYNC_MODEL` | Trained recogniser to load. Enables sign recognition. |
| `SIGNSYNC_BOOTSTRAP_DEMO` | `1` trains a synthetic demo model on first boot. Not USL. |
| `SIGNSYNC_LEXICON` | Replace the bundled placeholder lexicon with a reviewed one. |
| `SIGNSYNC_CLIPS` | Recorded motion library. Without it the avatar uses generated motion. |
| `SIGNSYNC_VOICE` | Piper voice for local speech output. |
| `SIGNSYNC_MIN_CONFIDENCE` | Below this the pipeline warns and asks the signer to repeat. |
| `SIGNSYNC_REQUIRE_MODEL` | `1` refuses to start without a recogniser. |

A path that does not exist **stops the container at startup**. Starting anyway would leave a
service that accepts sign input and recognises nothing, which from the clinic is indistinguishable
from a system that does not work — and takes far longer to diagnose than a container that refuses
to boot. `SIGNSYNC_REQUIRE_MODEL=1` extends that to sites which need Mode A; it is off by default
because the avatar-only Mode B deployments in plan §18.3 are legitimate on their own.

## Data handling

`data/` and `artifacts/` are **bind-mounted read-only**, never copied into the image. Image layers
are cached, pushed and pulled, which would put corpus video, landmark files and consent records in
places no one can reliably delete them from — incompatible with the withdrawal and retention
obligations in [docs/data-protection.md](../docs/data-protection.md). `.dockerignore` keeps them out
of the build context for the same reason.

The container runs unprivileged, with a read-only root filesystem and `no-new-privileges`. This
service processes video-derived data about identifiable people; that is not a workload to run as
root.

Serving reads a model and never writes one, so `artifacts/` is read-only too. If you want the
container to save models to the host, `--writable-artifacts` mounts it writable *and* runs the
container as your uid — a bind mount keeps the host's ownership, so without that the image's
unprivileged user cannot write to it.

`--demo` keeps its synthetic corpus and model inside the container, where they vanish on exit. That
is deliberate on top of the permissions point: a demo model is scaffolding, and one left sitting in
`artifacts/` is one somebody later mistakes for a trained system.

## Before a real deployment

Code cannot discharge these, and none of them are in this directory:

- [ ] PDPO registration as data controller, and a completed Data Protection Impact Assessment
      (plan §16.1).
- [ ] TLS termination. The compose file exposes plain HTTP because it assumes a trusted local
      network; anything reachable beyond one needs a reverse proxy with a certificate.
- [ ] Authentication. There is none. A deployment on a shared network needs it.
- [ ] A breach notification plan, in place *before* the first recording (plan §16.1).
- [ ] Agreement with UNAD and the site on who holds the data and for how long (plan §17,
      long-term custodianship).
- [ ] At least one certified human evaluation round — `signsync evaluate` will tell you it is
      missing, and it is right to block on it (plan §15).
