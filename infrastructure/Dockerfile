# SignSync API image.
#
# Two stages so the runtime layer carries no build toolchain. The base is
# python:3.12-slim rather than a CUDA image because plan §17 targets local-first
# deployment on modest hardware — the landmark-based models in §8.1 are chosen
# precisely so a clinic laptop can run this without a GPU, and shipping a
# multi-gigabyte CUDA image would contradict that.
#
# Optional extras are a build argument, and the default is the *smallest* set that
# serves both directions: `api` only. `vision`, `models` and `runtime` each pull in
# hundreds of megabytes, and none is needed by a deployment where browser clients
# send landmarks and the NumPy recogniser serves them. Plan §17 expects this image
# to reach sites over slow links, sometimes on a USB stick, so every extra is opt-in
# rather than opt-out.
#
#   docker build -f infrastructure/Dockerfile -t signsync .
#   docker build -f infrastructure/Dockerfile --build-arg EXTRAS=api,models -t signsync .
#   docker build -f infrastructure/Dockerfile --build-arg EXTRAS=api,runtime -t signsync .

FROM python:3.12-slim AS build

ARG EXTRAS="api"

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata first so the install layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[${EXTRAS}]"


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="SignSync" \
      org.opencontainers.image.description="Bidirectional Ugandan Sign Language <-> English translation" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Brair-Mpagi/SignSync"

# Runs unprivileged. This service handles video-derived landmark data from
# identifiable people (plan §16); a container running as root is not a reasonable
# place to process it.
RUN useradd --create-home --uid 10001 signsync

COPY --from=build /opt/venv /opt/venv
COPY --chmod=0755 infrastructure/entrypoint.sh /usr/local/bin/signsync-entrypoint

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIGNSYNC_HOST=0.0.0.0 \
    SIGNSYNC_PORT=8000

# Data and models are mounted, never baked in. Corpus video, landmarks and consent
# records must not end up in an image layer — layers are cached, pushed and pulled,
# which makes deletion-on-withdrawal impossible (docs/data-protection.md).
RUN mkdir -p /data /artifacts && chown signsync:signsync /data /artifacts
VOLUME ["/data", "/artifacts"]

USER signsync
WORKDIR /home/signsync
EXPOSE 8000

# Checks /health rather than the port, so a container that is listening but has no
# working pipeline is reported unhealthy rather than ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["signsync-entrypoint"]
