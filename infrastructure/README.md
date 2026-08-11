# infrastructure/

Deployment for the local-first strategy in [plan §17](../docs/plan.md).

## The shape this assumes

Plan §17 pilots in "a health clinic, a school for the Deaf, a government service desk". The
realistic deployment there is **one machine on a local network**, often without reliable internet,
sometimes without any. So:

- One container. The API also serves the browser client, so a pilot site runs a single service.
- No GPU, no CUDA base image. Landmark-based models (plan §8.1) are chosen so CPU inference is
  enough; a multi-gigabyte GPU image would contradict the reason for that choice.
- Optional extras are build arguments. The default image omits `vision` and `models`, because a
  deployment where browser clients send landmarks and the NumPy recogniser serves them needs
  neither, and every hundred megabytes matters when the image is copied over a slow link or
  sneakernetted on a USB stick.

## Run

```bash
docker compose -f infrastructure/docker-compose.yml up --build
# API and client: http://localhost:8000
```

With the React client (needs a package registry and a build step):

```bash
docker compose -f infrastructure/docker-compose.yml --profile frontend up
```

Building with the trainable models instead:

```bash
docker compose build --build-arg EXTRAS=api,models,runtime
```

## Data handling

`data/` and `artifacts/` are **bind-mounted read-only**, never copied into the image. Image layers
are cached, pushed and pulled, which would put corpus video, landmark files and consent records in
places no one can reliably delete them from — incompatible with the withdrawal and retention
obligations in [docs/data-protection.md](../docs/data-protection.md).

The container runs unprivileged with a read-only root filesystem and `no-new-privileges`. This
service processes video-derived data about identifiable people; that is not a workload to run as
root.

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
