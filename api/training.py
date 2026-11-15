"""Local training jobs. The worker survives UI reloads; each job owns its artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "logs" / "training_jobs"
router = APIRouter(prefix="/training", tags=["training"])
lock = threading.Lock()
processes: dict[str, subprocess.Popen] = {}


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def job_path(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Entrenamiento desconocido")
    path = JOBS / job_id
    if not (path / "state.json").exists():
        raise HTTPException(404, "Entrenamiento desconocido")
    return path


def checkpoint_for(job_id):
    state = read(job_path(job_id) / "state.json")
    run_id = state.get("run_id")
    if not run_id or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(409, "Todavía no hay pesos guardados")
    directory = ROOT / "models" / run_id
    if state["status"] in ("starting", "running", "stopping"):
        raise HTTPException(409, "Detén y guarda el entrenamiento antes de continuar o evaluar")
    hashes = read(directory / "sha256.json", {})
    for name in ("ppo_model.zip", "vecnormalize.pkl", "optimizer.pt", "checkpoint_state.json"):
        path = directory / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
            raise HTTPException(409, f"Checkpoint incompleto o alterado: {name}")
    config = read(directory / "config.json")
    if config["environment"].get("history_steps") != 4 or not config["environment"].get("campaign"):
        raise HTTPException(409, "Checkpoint incompatible con la campaña")
    return directory


def describe(path):
    state = read(path / "state.json")
    process = processes.get(path.name)
    if process and process.poll() is not None and state["status"] in ("starting", "running", "stopping"):
        state.update(status="failed", error=f"El proceso terminó con código {process.returncode}; consulta el registro")
        write(path / "state.json", state)
    elif not process and state["status"] in ("starting", "running", "stopping") and not worker_alive(state):
        state.update(status="failed", error="El proceso ya no está activo; los checkpoints guardados se conservan")
        write(path / "state.json", state)
    directory = ROOT / "models" / (state.get("run_id") or "unavailable")
    if (path / "stop").exists() and state["status"] in ("starting", "running"):
        state = {**state, "status": "stopping"}
    return {**state, "progress": read(directory / "progress.json", {}),
            "checkpoint": read(directory / "checkpoint_state.json"),
            "evaluation": read(directory / "evaluacion.json", {}).get("metrics"),
            "manifest": read(ROOT / "manifests" / f"{state.get('run_id')}.json"),
            "checkpoints": [p.name for p in sorted(directory.glob("checkpoint-*")) if p.is_dir()]}


def worker_alive(state):
    pid = state.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@router.get("/jobs")
def jobs():
    return {"enabled": os.environ.get("ENABLE_LOCAL_TRAINING") == "1",
            "jobs": [describe(p.parent) for p in sorted(JOBS.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)]}


class StartTraining(BaseModel):
    budget: int = Field(default=65536, ge=128, le=2000000)
    seed: int = Field(default=1, ge=0, le=2147483647)
    parent_job_id: str | None = None


@router.post("/jobs")
def start(req: StartTraining):
    if os.environ.get("ENABLE_LOCAL_TRAINING") != "1":
        raise HTTPException(403, "El entrenamiento debe habilitarse en el servidor local")
    campaign = read(ROOT / "logs/autonomous_campaign/state.json", {})
    if worker_alive(campaign) and campaign.get("status") in ("running", "validating", "stopping") and os.environ.get("EXPEDITION_CAMPAIGN_WORKER") != "1":
        raise HTTPException(409, "La campaña automática administra el entrenamiento; detén la campaña primero")
    with lock:
        if any(p.poll() is None for p in processes.values()) or any(
            worker_alive(read(p)) for p in JOBS.glob("*/state.json")
            if read(p).get("status") in ("starting", "running", "stopping")
        ):
            raise HTTPException(409, "Ya hay un entrenamiento activo")
        parent = checkpoint_for(req.parent_job_id) if req.parent_job_id else None
        job_id = uuid.uuid4().hex
        directory = JOBS / job_id
        config = yaml.safe_load((ROOT / "configs/experiments/exp-010-parallel.yaml").read_text(encoding="utf-8"))
        config.update(budget_timesteps=req.budget, seeds=[req.seed],
                      parent_run_id=f"local:{parent}" if parent else None)
        config["hyperparams"]["n_steps"] = 256
        config["environment"]["episode_db"] = str(ROOT / "logs/planetary_episodes.sqlite")
        state = {"job_id": job_id, "status": "starting", "algorithm": "PPO",
                 "budget": req.budget, "seed": req.seed, "parent_job_id": req.parent_job_id,
                 "run_id": None}
        write(directory / "state.json", state)
        write(directory / "config.json", config)
        with (directory / "worker.log").open("wb") as log:
            processes[job_id] = subprocess.Popen(
                [sys.executable, "-m", "api.training_worker", str(directory)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                     "EXPEDITION_STOP_FILE": str(directory / "stop"),
                     "MLFLOW_TRACKING_URI": f"sqlite:///{ROOT / 'mlflow.db'}",
                     "MLFLOW_REGISTRY_URI": f"sqlite:///{ROOT / 'mlflow.db'}"},
            )
        # Publish the PID before the worker finishes importing torch/MLflow,
        # so another API process does not mistake its startup for a crash.
        state = read(directory / "state.json")
        state["pid"] = processes[job_id].pid
        write(directory / "state.json", state)
        return state


@router.post("/jobs/{job_id}/stop")
def stop(job_id: str):
    if os.environ.get("ENABLE_LOCAL_TRAINING") != "1":
        raise HTTPException(403, "Entrenamiento local deshabilitado")
    directory = job_path(job_id)
    (directory / "stop").touch()
    return {"status": "stop_requested", "detail": "Se guardará al terminar la actualización PPO actual"}


@router.get("/jobs/{job_id}/log")
def log(job_id: str):
    path = job_path(job_id) / "worker.log"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 12000))
        return {"text": stream.read().decode("utf-8", errors="replace")}


@router.get("/campaign")
def campaign_status():
    state = read(ROOT / "logs/autonomous_campaign/state.json", {"status": "not_started"})
    if state.get("status") in ("running", "validating", "stopping") and not worker_alive(state):
        state = {**state, "status": "offline"}
    return {**state, "release": read(ROOT / "models/final-release.json")}


@router.post("/campaign/stop")
def stop_campaign():
    if os.environ.get("ENABLE_LOCAL_TRAINING") != "1":
        raise HTTPException(403, "Entrenamiento local deshabilitado")
    directory = ROOT / "logs/autonomous_campaign"
    state = read(directory / "state.json", {})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stop").touch()
    if state.get("active_job_id"):
        stop(state["active_job_id"])
    return {"status": "stop_requested"}
