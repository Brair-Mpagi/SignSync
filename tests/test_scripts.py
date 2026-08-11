"""The shell entry points.

`run.sh` and the container entrypoint are the paths a contributor and a pilot site
actually take, so their argument handling and their guards deserve the same
treatment as the Python. The entrypoint tests run it with stub executables on PATH,
which exercises the real control flow without needing Docker or a built image.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_SH = REPO / "run.sh"
RUN_DOCKER = REPO / "infrastructure" / "run-docker.sh"
ENTRYPOINT = REPO / "infrastructure" / "entrypoint.sh"


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=60, **kwargs)


# --------------------------------------------------------------------------- presence


@pytest.mark.parametrize("script", [RUN_SH, RUN_DOCKER, ENTRYPOINT])
def test_scripts_exist_and_are_executable(script):
    assert script.is_file(), f"{script.name} is missing"
    assert os.access(script, os.X_OK), f"{script.name} is not executable"


@pytest.mark.parametrize("script", [RUN_SH, RUN_DOCKER])
def test_scripts_parse(script):
    assert run(["bash", "-n", str(script)]).returncode == 0


def test_entrypoint_is_posix_sh():
    """The image has no bash, so the entrypoint must not need one."""
    assert run(["sh", "-n", str(ENTRYPOINT)]).returncode == 0
    assert ENTRYPOINT.read_text(encoding="utf-8").startswith("#!/usr/bin/env sh")


# --------------------------------------------------------------------------- run.sh


def test_run_help_lists_the_stages():
    result = run([str(RUN_SH), "--help"])
    assert result.returncode == 0
    for stage in ("setup", "corpus", "train", "evaluate", "serve"):
        assert stage in result.stdout


def test_run_rejects_an_unknown_option():
    result = run([str(RUN_SH), "--wat"])
    assert result.returncode != 0
    assert "unknown option" in result.stderr


def test_run_rejects_an_unknown_stage():
    result = run([str(RUN_SH), "--only", "banana"])
    assert result.returncode != 0
    assert "unknown stage" in result.stderr


def test_only_restricts_to_one_stage():
    """`--only corpus` must not also run the demo or start a server."""
    source = RUN_SH.read_text(encoding="utf-8")
    only_block = source.split("--only)")[1].split("shift ;;")[0]
    for flag in ("DO_SETUP=0", "DO_CORPUS=0", "DO_TRAIN=0", "DO_EVALUATE=0", "DO_SERVE=0", "DO_DEMO=0"):
        assert flag in only_block, f"--only does not clear {flag}"


def test_run_docker_help_documents_the_model_options():
    result = run([str(RUN_DOCKER), "--help"])
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--demo" in result.stdout


def test_run_docker_rejects_an_unknown_option():
    result = run([str(RUN_DOCKER), "--nope"])
    assert result.returncode != 0
    assert "unknown option" in result.stderr


# --------------------------------------------------------------------------- entrypoint


@pytest.fixture
def stub_bin(tmp_path):
    """A PATH where `signsync` and `uvicorn` are recorded rather than executed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"

    # The signsync stub honours `train --out <path>` by touching the file, so the
    # entrypoint's "did training actually produce a model?" check sees what it would
    # see in a real container.
    signsync = bin_dir / "signsync"
    signsync.write_text(
        f'''#!/usr/bin/env sh
echo "signsync $*" >> "{log}"
prev=""
for arg in "$@"; do
  if [ "$prev" = "--out" ]; then mkdir -p "$(dirname "$arg")"; : > "$arg"; fi
  prev="$arg"
done
exit 0
''',
        encoding="utf-8",
    )
    signsync.chmod(0o755)

    uvicorn = bin_dir / "uvicorn"
    uvicorn.write_text(f'#!/usr/bin/env sh\necho "uvicorn $*" >> "{log}"\nexit 0\n', encoding="utf-8")
    uvicorn.chmod(0o755)

    return bin_dir, log


def run_entrypoint(stub_bin, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the entrypoint with stubs on PATH.

    The bootstrap paths are redirected into the test's own directory. Their real
    defaults are fixed paths under /tmp, which is correct in a container with a
    fresh filesystem but would let one test's leftovers satisfy the next one's
    check when they all share the host's /tmp.
    """
    bin_dir, log = stub_bin
    sandbox = bin_dir.parent
    full_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SIGNSYNC_BOOTSTRAP_CORPUS": str(sandbox / "corpus"),
        "SIGNSYNC_BOOTSTRAP_MODEL": str(sandbox / "demo.npz"),
        **env,
    }
    result = run(["sh", str(ENTRYPOINT)], env=full_env)
    return result, log.read_text(encoding="utf-8") if log.exists() else ""


def test_entrypoint_refuses_a_model_path_that_is_not_there(stub_bin, tmp_path):
    """Starting anyway leaves a service that accepts signing and recognises nothing."""
    result, calls = run_entrypoint(stub_bin, {"SIGNSYNC_MODEL": str(tmp_path / "absent.npz")})

    assert result.returncode != 0
    assert "not in the container" in result.stderr
    assert "uvicorn" not in calls, "the server started despite a missing model"


def test_entrypoint_starts_without_a_model(stub_bin):
    """Avatar-only deployments (plan §18.3 Mode B) are legitimate."""
    result, calls = run_entrypoint(stub_bin, {})
    assert result.returncode == 0
    assert "uvicorn signsync.api:app" in calls


def test_entrypoint_accepts_a_model_that_exists(stub_bin, tmp_path):
    model = tmp_path / "model.npz"
    model.write_bytes(b"")
    result, calls = run_entrypoint(stub_bin, {"SIGNSYNC_MODEL": str(model)})
    assert result.returncode == 0
    assert "uvicorn" in calls


def test_demo_bootstrap_trains_and_stays_out_of_the_mounted_volume(stub_bin):
    """A demo model in artifacts/ is one somebody later mistakes for a trained system."""
    result, calls = run_entrypoint(stub_bin, {"SIGNSYNC_BOOTSTRAP_DEMO": "1"})

    assert result.returncode == 0
    assert "corpus build-synthetic" in calls
    assert "train" in calls
    assert "/artifacts" not in calls, "demo bootstrap wrote to the mounted volume"


def test_demo_bootstrap_says_the_model_is_not_usl(stub_bin):
    result, _ = run_entrypoint(stub_bin, {"SIGNSYNC_BOOTSTRAP_DEMO": "1"})
    assert "not trained on Ugandan Sign Language" in result.stdout


def test_a_failed_bootstrap_is_reported_as_a_bootstrap_failure(stub_bin, tmp_path):
    """Not as "mount a model" — which would send the operator down the wrong path."""
    bin_dir, log = stub_bin
    broken = bin_dir / "signsync"
    broken.write_text(f'#!/usr/bin/env sh\necho "signsync $*" >> "{log}"\nexit 0\n', encoding="utf-8")
    broken.chmod(0o755)

    result, calls = run_entrypoint(stub_bin, {"SIGNSYNC_BOOTSTRAP_DEMO": "1"})

    assert result.returncode != 0
    assert "demo bootstrap did not produce a model" in result.stderr
    assert "uvicorn" not in calls


def test_an_explicit_model_wins_over_the_demo_bootstrap(stub_bin, tmp_path):
    model = tmp_path / "real.npz"
    model.write_bytes(b"")
    _, calls = run_entrypoint(
        stub_bin, {"SIGNSYNC_MODEL": str(model), "SIGNSYNC_BOOTSTRAP_DEMO": "1"}
    )
    assert "build-synthetic" not in calls, "a supplied model was overwritten by the demo"


def test_host_and_port_are_configurable(stub_bin):
    _, calls = run_entrypoint(stub_bin, {"SIGNSYNC_HOST": "127.0.0.1", "SIGNSYNC_PORT": "9001"})
    assert "--host 127.0.0.1" in calls
    assert "--port 9001" in calls


# --------------------------------------------------------------------------- docker assets


def test_dockerignore_excludes_participant_data():
    """Image layers are cached and pushed; anything in one is effectively undeletable."""
    ignored = (REPO / ".dockerignore").read_text(encoding="utf-8")
    for pattern in ("data/", "artifacts/", "*.npz", ".git/"):
        assert pattern in ignored, f".dockerignore is missing {pattern}"


def test_dockerfile_runs_unprivileged_and_does_not_bake_in_data():
    dockerfile = (REPO / "infrastructure" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER signsync" in dockerfile
    assert "useradd" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_default_image_does_not_bake_in_the_heavy_extras():
    """Plan §17: this image travels over slow links, so extras are opt-in."""
    dockerfile = (REPO / "infrastructure" / "Dockerfile").read_text(encoding="utf-8")
    default = next(line for line in dockerfile.splitlines() if line.startswith("ARG EXTRAS"))
    for heavy in ("models", "vision", "runtime"):
        assert heavy not in default, f"default image includes the {heavy} extra"
