# SignSync API image.
#
# Two stages so the runtime layer carries no build toolchain. The base is
# python:3.12-slim rather than a CUDA image because plan §17 targets local-first
# deployment on modest hardware — the landmark-based models in §8.1 are chosen
# precisely so a clinic laptop can run this without a GPU, and shipping a
# multi-gigabyte CUDA image would contradict that.
#
# Optional extras are build arguments. The default image includes the API and the
# runtime (ONNX) extras only: `vision` and `models` pull in hundreds of megabytes,
# and a deployment that receives landmarks from browser clients and runs the NumPy
# recogniser needs neither.

FROM python:3.12-slim AS build

ARG EXTRAS="api,runtime"

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[${EXTRAS}]"


FROM python:3.12-slim AS runtime

# Runs unprivileged. This service handles video-derived landmark data from
# identifiable people (plan §16); a container running as root is not a reasonable
# place to process it.
RUN useradd --create-home --uid 10001 signsync

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Data is mounted, never baked in. Corpus video, landmarks and consent records must
# not end up in an image layer — layers are copied, cached and pushed, which makes
# deletion-on-withdrawal impossible (docs/data-protection.md).
VOLUME ["/data"]

USER signsync
WORKDIR /home/signsync
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "signsync.api:app", "--host", "0.0.0.0", "--port", "8000"]
