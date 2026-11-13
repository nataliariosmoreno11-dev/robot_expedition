"""API de simulacion y evaluacion de Robot Expedition (paso 4 del plan).

Usa el MISMO ambiente Python (MuJoCo) y las mismas politicas que el
entrenamiento y la suite de evaluacion (pipelines/), asi la web evalua los
modelos con sus reglas originales.

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("API_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.store import LEVELS, ModelIntegrityError, ModelUnavailable, RunNotFound, Store  # noqa: E402
from pipelines.evaluate import frame, load_suite, run_case, summarize  # noqa: E402
from robot_expedition.envs.beacon_expedition import (  # noqa: E402
    ACTIONS, ENVIRONMENT_VERSION, BeaconExpeditionEnv, environment_hash,
)
from robot_expedition.envs.planetary_exploration import (  # noqa: E402
    PLANETARY_ACTIONS,
    PLANETARY_ENVIRONMENT_VERSION,
    PlanetaryExplorationEnv,
)
from robot_expedition.planetary.episode_store import EpisodeStore  # noqa: E402

API_VERSION = "1.1.0"
MAX_SEEDS = 20                 # la web corta las solicitudes a los 12 s
MAX_SESSIONS = 50
SESSION_TTL_S = 15 * 60
STARTED = time.time()
EPISODE_DB = Path(os.environ.get("EXPEDITION_EPISODE_DB", ROOT / "logs" / "planetary_episodes.sqlite"))

store = Store(ROOT)
app = FastAPI(title="Robot Expedition API", version=API_VERSION,
              description="Simulación 3D (MuJoCo) y evaluación de modelos de Robot Expedition.")

DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:4173,https://web-five-gules-83.vercel.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# ---------------------------------------------------------------- monitoreo
class Monitor:
    """Contadores en memoria: solicitudes, errores y latencia por ruta."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = defaultdict(int)
        self.errors = defaultdict(int)
        self.latency_ms = defaultdict(lambda: deque(maxlen=200))
        self.last_errors = deque(maxlen=20)

    def record(self, route: str, status: int, ms: float, detail: str | None = None):
        with self.lock:
            self.requests[route] += 1
            self.latency_ms[route].append(ms)
            if status >= 400:
                self.errors[route] += 1
                self.last_errors.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "route": route, "status": status, "detail": detail})

    def snapshot(self) -> dict:
        with self.lock:
            routes = {}
            for route, count in self.requests.items():
                lat = sorted(self.latency_ms[route])
                routes[route] = {"requests": count, "errors": self.errors[route],
                                 "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
                                 "latency_ms_p95": round(lat[max(0, math.ceil(0.95 * len(lat)) - 1)], 2) if lat else None}
            return {"routes": routes, "last_errors": list(self.last_errors)}


monitor = Monitor()


@app.middleware("http")
async def measure(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.method == "OPTIONS":                 # preflight CORS: no es trafico de la aplicacion
        return response
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    monitor.record(f"{request.method} {path}", response.status_code, (time.perf_counter() - t0) * 1000)
    return response


def fail(status: int, detail: str):
    raise HTTPException(status_code=status, detail=detail)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def policy_for(run_id: str):
    try:
        return store.policy(run_id)
    except RunNotFound:
        fail(404, f"no existe el run {run_id}")
    except ModelIntegrityError as exc:
        fail(409, str(exc))
    except ModelUnavailable as exc:
        fail(503, str(exc))


# ------------------------------------------------------------------ estado
@app.get("/")
def root():
    """Portada de la API: que es y donde esta cada cosa (la raiz no es un 404)."""
    return {
        "service": "Robot Expedition API",
        "description": "Simulación 3D (MuJoCo) y evaluación de modelos del proyecto final de MLOps.",
        "web": "https://web-five-gules-83.vercel.app",
        "docs": "/docs",
        "routes": ["/health", "/version", "/catalog", "/runs/{run_id}", "/evaluate",
                   "/sim/reset", "/sim/step", "/episodes", "/episodes/{number}",
                   "/episodes/{number}/replay", "/episodes/{number}/scenario",
                   "/episodes/{number}/protect", "/episodes/metrics/summary",
                   "/episodes/compare", "/monitor"],
        "repository": "https://github.com/nataliariosmoreno11-dev/robot_expedition",
    }


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED, 1), "sessions": len(sessions)}


@app.get("/version")
def version():
    catalog = store.catalog()
    return {
        "api_version": API_VERSION,
        "git_commit": os.environ.get("GIT_COMMIT") or os.environ.get("RENDER_GIT_COMMIT") or _git("rev-parse", "HEAD"),
        "release": os.environ.get("RELEASE") or _git("describe", "--tags", "--always"),
        "environment_version": ENVIRONMENT_VERSION,
        "planetary_environment_version": PLANETARY_ENVIRONMENT_VERSION,
        "environment_hashes": {level: environment_hash(level) for level in LEVELS},
        "served_version": catalog.get("served_version"),
        "recommended_run_id": catalog.get("recommended_run_id"),
        "catalog_source": catalog.get("source"),
        "python": sys.version.split()[0],
        "policy_backend": os.environ.get("POLICY_BACKEND", "sb3"),
    }


@app.get("/monitor")
def monitor_view():
    return {**monitor.snapshot(), "sessions": len(sessions), "uptime_s": round(time.time() - STARTED, 1)}


@app.get("/catalog")
def catalog():
    return store.catalog()


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    try:
        return store.run(run_id)
    except RunNotFound:
        fail(404, f"no existe el run {run_id}")


# --------------------------------------------------------------- evaluar
class EvaluateRequest(BaseModel):
    run_id: str
    level_id: Literal["n1", "n2", "n3"]
    seeds: list[int] | None = Field(default=None, description=f"hasta {MAX_SEEDS} semillas")
    suite_id: str | None = Field(default=None, description="alternativa a seeds: suite fija (configs/suites)")
    episodes: int | None = Field(default=None, ge=1, le=50, description="limita la suite")


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    run = run_detail(req.run_id)
    if run.get("environment_version", ENVIRONMENT_VERSION).split("/")[0] != ENVIRONMENT_VERSION.split("/")[0]:
        fail(409, f"el run usa {run.get('environment_version')}, incompatible con {ENVIRONMENT_VERSION}")
    environment = {}
    if req.seeds:
        seeds, suite_id = list(dict.fromkeys(req.seeds)), None
        if len(seeds) > MAX_SEEDS:
            fail(422, f"máximo {MAX_SEEDS} semillas por solicitud; usa suite_id para la suite completa")
    else:
        suite_id = req.suite_id or f"suite-{req.level_id}-v1"
        try:
            suite = load_suite(suite_id)
        except FileNotFoundError:
            fail(404, f"no existe la suite {suite_id}")
        except RuntimeError as exc:
            fail(403, str(exc))
        if suite.get("level_id") not in (None, req.level_id):
            fail(422, f"la suite {suite_id} es del nivel {suite['level_id']}")
        seeds = suite["seeds"][:req.episodes] if req.episodes else suite["seeds"]
        environment = dict(suite.get("environment") or {})
        environment.setdefault("max_steps", int(suite.get("max_steps", 400)))
    policy = policy_for(req.run_id)

    env = BeaconExpeditionEnv(level=req.level_id, **environment)
    started = time.perf_counter()
    frames: list = []
    cases = [run_case(env, policy, seeds[0], frames)]
    first_map = env.map_spec()
    first_rocks = [list(r) for r in env.rocks]
    first_blocking = env.episode_blocking
    cases.extend(run_case(env, policy, seed) for seed in seeds[1:])
    first = cases[0]
    episode = {
        "episode": first["seed"], "mode": "evaluación", "run_id": req.run_id,
        "algorithm": run.get("algorithm"), "model_name": run.get("model_name") or run.get("algorithm"),
        "environment_version": ENVIRONMENT_VERSION, "level_id": req.level_id,
        "scenario_id": first["scenario_id"], "difficulty": first["difficulty"],
        "frames": frames, "rocks": first_rocks, "block": first_blocking,
        "success": first["success"], "end": first["end"], "reward": first["return"],
        "steps": first["steps"], "visited": first["visited"], "battery": first["battery"],
        "path_length": first["path_length"], "latency_ms": first["latency_ms_mean"],
    }
    result = {
        "run_id": req.run_id, "level_id": req.level_id, "algorithm": run.get("algorithm"),
        "trained_level_id": run.get("level_id"), "model_sha256": run.get("model_sha256"),
        "environment_version": ENVIRONMENT_VERSION,
        "environment_hash": environment_hash(req.level_id, environment), "environment": environment,
        "suite_id": suite_id, "seeds": seeds, "deterministic": True,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "map": first_map, "metrics": summarize(cases, env.beacon_count),
        "cases": cases, "episode": episode,
    }
    env.close()
    return result


# ------------------------------------------------------- control manual
class Session:
    def __init__(self, level_id: str, policy, policy_id: str, seed: int,
                 environment_kind: str = "beacon", difficulty_level: int = 1):
        if environment_kind == "planetary":
            self.env = PlanetaryExplorationEnv(
                difficulty_level=difficulty_level,
                curriculum=False,
                episode_db=EPISODE_DB,
                record_episodes=True,
                run_id="api-manual" if policy is None else policy_id,
                policy_version=policy_id,
                mode="manual" if policy is None else "evaluación",
                update_curriculum=False,
            )
            self.action_names = [item[0] for item in PLANETARY_ACTIONS]
        else:
            self.env = BeaconExpeditionEnv(level=level_id)
            self.action_names = [item[0] for item in ACTIONS]
        self.policy, self.policy_id, self.level_id = policy, policy_id, level_id
        self.environment_kind = environment_kind
        self.obs, _ = self.env.reset(seed=seed)
        self.done, self.total, self.touched = False, 0.0, time.time()


sessions: OrderedDict[str, Session] = OrderedDict()
sessions_lock = threading.Lock()


def _prune_sessions():
    now = time.time()
    for sid in [sid for sid, s in sessions.items() if now - s.touched > SESSION_TTL_S]:
        sessions.pop(sid).env.close()
    while len(sessions) >= MAX_SESSIONS:
        sessions.popitem(last=False)[1].env.close()


class ResetRequest(BaseModel):
    level_id: Literal["n1", "n2", "n3"] = "n2"
    seed: int = 0
    policy: str = Field(default="manual", description="manual, baseline-rule, baseline-random o un run_id")
    environment: Literal["beacon", "planetary"] = "beacon"
    difficulty_level: int = Field(default=1, ge=1, le=8)


class StepRequest(BaseModel):
    session_id: str
    action: int | None = Field(default=None, ge=0, le=len(PLANETARY_ACTIONS) - 1,
                               description="acción manual; vacía = decide la política de la sesión")


@app.post("/sim/reset")
def sim_reset(req: ResetRequest):
    if req.environment == "planetary" and req.policy != "manual":
        fail(422, "los modelos publicados usan el contrato anterior; usa manual o entrena exp-007-planetary")
    policy = None if req.policy == "manual" else policy_for(req.policy)
    session = Session(
        req.level_id,
        policy,
        req.policy,
        req.seed,
        req.environment,
        req.difficulty_level,
    )
    sid = uuid.uuid4().hex
    with sessions_lock:
        _prune_sessions()
        sessions[sid] = session
    env = session.env
    return {"session_id": sid, "mode": "manual" if policy is None else "política", "policy": req.policy,
            "level_id": req.level_id, "seed": req.seed,
            "environment": req.environment,
            "environment_version": (PLANETARY_ENVIRONMENT_VERSION if req.environment == "planetary"
                                    else ENVIRONMENT_VERSION),
            "map": env.map_spec(), "frame": frame(env, -1, 0.0, False), "actions": session.action_names,
            "terminated": False, "truncated": False, "end": "running"}


@app.post("/sim/step")
def sim_step(req: StepRequest):
    with sessions_lock:
        session = sessions.get(req.session_id)
        if session is not None:
            sessions.move_to_end(req.session_id)
    if session is None:
        fail(404, "la sesión no existe o expiró; inicia otra con /sim/reset")
    if session.done:
        fail(409, "el episodio de esta sesión ya terminó; inicia otro con /sim/reset")
    if req.action is None and session.policy is None:
        fail(422, f"sesión manual: envía action (0-{len(session.action_names) - 1})")
    action = req.action if req.action is not None else int(session.policy(session.obs))
    if not session.env.action_space.contains(action):
        fail(422, f"acción inválida para {session.environment_kind}: {action}")
    session.obs, reward, terminated, truncated, info = session.env.step(action)
    session.total += reward
    session.done = terminated or truncated
    session.touched = time.time()
    return {"session_id": req.session_id, "frame": frame(session.env, action, reward, info["collision"]),
            "action": action, "action_name": session.action_names[action], "reward": round(float(reward), 4),
            "return": round(session.total, 4), "terminated": terminated, "truncated": truncated,
            "end": info["end"] or "running", "event": info["event"], "stage": info["stage"],
            "battery": round(info["battery"], 2), "collisions": info["collisions"],
            "success": info["success"], "steps": session.env.steps,
            "target_type": info.get("target_type"),
            "requested_samples": info.get("requested_samples"),
            "samples_collected": info.get("samples_collected"),
            "inventory": info.get("inventory")}


# ------------------------------------------------------- historial planetario
def _with_episode_store(callback):
    store = EpisodeStore(EPISODE_DB)
    try:
        return callback(store)
    finally:
        store.close()


@app.get("/episodes")
def episodes(limit: int = 100, offset: int = 0, run_id: str | None = None):
    return {
        "schema_version": 2,
        "database": EPISODE_DB.name,
        "storage": _with_episode_store(lambda episode_store: episode_store.storage_status()),
        "episodes": _with_episode_store(
            lambda episode_store: episode_store.list_episodes(limit=limit, offset=offset, run_id=run_id)
        ),
    }


@app.get("/episodes/metrics/summary")
def episode_metrics(run_id: str | None = None):
    return _with_episode_store(lambda episode_store: episode_store.metrics(run_id=run_id))


class ProtectEpisodeRequest(BaseModel):
    protected: bool = True


@app.post("/episodes/{episode_number}/protect")
def protect_episode(episode_number: int, req: ProtectEpisodeRequest):
    try:
        return _with_episode_store(
            lambda episode_store: episode_store.protect_episode(episode_number, req.protected)
        )
    except KeyError as exc:
        fail(404, str(exc))
    except ValueError as exc:
        fail(409, str(exc))


@app.get("/episodes/{episode_number}")
def episode_data(episode_number: int):
    try:
        return _with_episode_store(lambda episode_store: episode_store.get_episode(episode_number))
    except KeyError as exc:
        fail(404, str(exc))


@app.get("/episodes/{episode_number}/replay")
def episode_replay(episode_number: int):
    try:
        return _with_episode_store(lambda episode_store: episode_store.replay_episode(episode_number))
    except KeyError as exc:
        fail(404, str(exc))


@app.get("/episodes/{episode_number}/scenario")
def episode_scenario(episode_number: int):
    try:
        return _with_episode_store(lambda episode_store: episode_store.load_scenario(episode_number))
    except KeyError as exc:
        fail(404, str(exc))


class CompareEpisodesRequest(BaseModel):
    episode_numbers: list[int] = Field(min_length=2, max_length=20)


@app.post("/episodes/compare")
def compare_episode_data(req: CompareEpisodesRequest):
    try:
        return _with_episode_store(
            lambda episode_store: episode_store.compare_episodes(*req.episode_numbers)
        )
    except KeyError as exc:
        fail(404, str(exc))
    except ValueError as exc:
        fail(422, str(exc))
