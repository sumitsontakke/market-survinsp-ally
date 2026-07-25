"""Launch Phase G stages as detached Docker containers from inside Streamlit.

Uses the Python ``docker`` SDK to invoke the trainer / trainer-gpu images on
the host daemon. The daemon resolves bind-mount paths on the host filesystem,
so the helper builds paths from the ``HOST_REPO_PATH`` env var (which compose
populates from ``${PWD}/..`` at startup).

Architecture:
  * Each stage launches a NEW detached container (auto-remove on exit).
  * Stage container IDs are written to /data/phase_g_jobs/<stage>.json so the
    page can poll status / stream logs across reloads.
  * Logs are streamed via container.logs(stream=True) and tail-buffered.
  * If the docker socket isn't reachable, ``available()`` returns False and
    the page falls back to copy-paste mode (the original UX).

Lifecycle helpers exposed to the page:
  * available()  -> bool
  * launch(stage) -> JobHandle
  * status(stage) -> dict
  * logs_tail(stage, n=200) -> str
  * cancel(stage) -> bool
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import docker  # type: ignore
    from docker.errors import NotFound, APIError, DockerException  # type: ignore
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    DockerException = Exception  # noqa: N816

JOB_DIR = Path("/data/phase_g_jobs")
HOST_REPO = os.environ.get("HOST_REPO_PATH", "").rstrip("/").rstrip("\\")

# Mapping of stage -> (image, command, gpu, label).
# Command is a list ready to pass to docker run.
STAGE_DEFS: dict[str, dict[str, Any]] = {
    "cohort": {
        "image": "nse-calibration-trainer:latest",
        "cmd":   ["python", "/app/scripts/run_phase_g.py", "--stage", "cohort"],
        "gpu":   False,
        "label": "Stage 1 — generate cohort plans",
    },
    "synth": {
        "image": "nse-calibration-trainer:latest",
        "cmd":   ["python", "/app/scripts/run_phase_g.py", "--stage", "synth"],
        "gpu":   False,
        "label": "Stage 2 — synthesize runs",
    },
    "train": {
        "image": "nse-calibration-trainer-gpu:latest",
        "cmd":   ["python", "-u", "/app/scripts/run_phase_g.py",
                  "--stage", "train"],
        "gpu":   True,
        "label": "Stage 3 — continual training",
    },
    "eval": {
        "image": "nse-calibration-trainer-gpu:latest",
        "cmd":   ["python", "-u", "/app/scripts/run_phase_g.py",
                  "--stage", "eval"],
        "gpu":   True,
        "label": "Stage 4 — OOD evaluation",
    },
}


@dataclass
class JobHandle:
    stage:        str
    container_id: str
    started_at:   float
    image:        str
    cmd:          list[str]
    state:        str = "running"     # running | done | failed | cancelled
    exit_code:    Optional[int] = None
    finished_at:  Optional[float] = None
    error:        str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobHandle":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _job_file(stage: str) -> Path:
    return JOB_DIR / f"{stage}.json"


def _save_job(j: JobHandle) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    _job_file(j.stage).write_text(json.dumps(j.to_dict(), indent=2),
                                  encoding="utf-8")


def _load_job(stage: str) -> Optional[JobHandle]:
    p = _job_file(stage)
    if not p.is_file():
        return None
    try:
        return JobHandle.from_dict(json.loads(p.read_text("utf-8")))
    except Exception:  # noqa: BLE001
        return None


def _client() -> Optional["docker.DockerClient"]:
    if not _HAS_SDK:
        return None
    try:
        c = docker.from_env()
        c.ping()
        return c
    except DockerException:
        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Bind-mount construction (mirrors compose service definitions exactly)
# ---------------------------------------------------------------------------
def _trainer_mounts(gpu: bool) -> dict[str, dict[str, str]]:
    """Volumes for trainer / trainer-gpu — must match the compose definitions
    so containers see the same paths as `docker compose run` would.
    """
    if not HOST_REPO:
        raise RuntimeError(
            "HOST_REPO_PATH env var not set on webapp-v2 service. "
            "Re-create the container so the env var picks up."
        )
    return {
        "nse_market_data": {"bind": "/data",        "mode": "rw"},
        f"{HOST_REPO}/calibration_service/core": {"bind": "/app/core",     "mode": "rw"},
        f"{HOST_REPO}/training":                  {"bind": "/app/training", "mode": "rw"},
        f"{HOST_REPO}/outputs":                   {"bind": "/app/outputs",  "mode": "rw"},
        f"{HOST_REPO}/scripts":                   {"bind": "/app/scripts",  "mode": "rw"},
        f"{HOST_REPO}/src":                       {"bind": "/app/src",      "mode": "rw"},
    }


def _trainer_env() -> dict[str, str]:
    return {
        "DB_PATH":            "/data/market.db",
        "PYTHONPATH":         "/app:/app/src",
        "OUTPUTS_PATH":       "/app/outputs",
        "MODEL_REGISTRY_DIR": "/data/model_registry",
        "TZ":                 "UTC",
    }


def _gpu_device_request():
    """Equivalent to compose's `deploy.resources.reservations.devices` GPU
    request. Requires nvidia-container-toolkit on the host.
    """
    if not _HAS_SDK:
        return None
    return docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def available() -> tuple[bool, str]:
    """Returns (can_run, reason). reason explains why if can_run is False."""
    if not _HAS_SDK:
        return False, ("Python docker SDK not installed in the webapp image. "
                       "`pip install docker` and rebuild the webapp image.")
    if not HOST_REPO:
        return False, ("HOST_REPO_PATH not set. Edit "
                       "calibration_service/docker-compose.yml and re-create "
                       "the webapp-v2 container.")
    c = _client()
    if c is None:
        return False, ("Docker socket /var/run/docker.sock not reachable "
                       "from inside the webapp container. Check that the "
                       "mount is in calibration_service/docker-compose.yml.")
    return True, "Docker daemon reachable; SDK ready."


def launch(stage: str) -> JobHandle:
    """Launch a Phase G stage. Returns a JobHandle persisted to /data/."""
    if stage not in STAGE_DEFS:
        raise ValueError(f"unknown stage '{stage}'")
    defn = STAGE_DEFS[stage]
    c = _client()
    if c is None:
        raise RuntimeError("Docker not reachable")

    existing = _load_job(stage)
    if existing and existing.state == "running":
        try:
            cont = c.containers.get(existing.container_id)
            if cont.status == "running":
                return existing
        except (NotFound, APIError):
            pass

    kwargs: dict[str, Any] = {
        "image":   defn["image"],
        "command": defn["cmd"],
        "volumes": _trainer_mounts(defn["gpu"]),
        "environment": _trainer_env(),
        "detach":  True,
        "auto_remove": False,
        "name":    f"phase_g_{stage}_{int(time.time())}",
    }
    if defn["gpu"]:
        kwargs["device_requests"] = [_gpu_device_request()]

    cont = c.containers.run(**kwargs)
    handle = JobHandle(
        stage=stage,
        container_id=cont.id,
        started_at=time.time(),
        image=defn["image"],
        cmd=list(defn["cmd"]),
        state="running",
    )
    _save_job(handle)
    return handle


def status(stage: str) -> Optional[JobHandle]:
    """Refresh the saved JobHandle from the daemon's view of the container."""
    h = _load_job(stage)
    if h is None:
        return None
    if h.state != "running":
        return h
    c = _client()
    if c is None:
        return h
    try:
        cont = c.containers.get(h.container_id)
    except NotFound:
        h.state = "done" if h.exit_code in (0, None) else h.state
        h.finished_at = h.finished_at or time.time()
        _save_job(h)
        return h
    if cont.status == "exited":
        info = cont.attrs.get("State", {})
        h.exit_code = int(info.get("ExitCode", 0))
        h.state = "done" if h.exit_code == 0 else "failed"
        h.finished_at = time.time()
        _save_job(h)
        try:
            cont.remove()
        except Exception:  # noqa: BLE001
            pass
    return h


def logs_tail(stage: str, n: int = 200) -> str:
    h = _load_job(stage)
    if h is None:
        return "(no job started yet)"
    c = _client()
    if c is None:
        return "(docker daemon not reachable)"
    try:
        cont = c.containers.get(h.container_id)
        raw = cont.logs(tail=n, stdout=True, stderr=True)
        return raw.decode("utf-8", errors="replace")
    except NotFound:
        return "(container no longer present — likely auto-removed)"
    except Exception as exc:  # noqa: BLE001
        return f"(error fetching logs: {exc!r})"


def cancel(stage: str) -> bool:
    h = _load_job(stage)
    if h is None:
        return False
    c = _client()
    if c is None:
        return False
    try:
        cont = c.containers.get(h.container_id)
        cont.stop(timeout=10)
        h.state = "cancelled"
        h.finished_at = time.time()
        _save_job(h)
        try:
            cont.remove()
        except Exception:  # noqa: BLE001
            pass
        return True
    except NotFound:
        h.state = "cancelled"
        _save_job(h)
        return True
    except Exception:  # noqa: BLE001
        return False
