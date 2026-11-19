"""API de simulacion y evaluacion de Robot Expedition (paso 4 del plan).

Usa el MISMO ambiente Python (MuJoCo) y las mismas politicas que el
entrenamiento y la suite de evaluacion (pipelines/), asi la web evalua los
modelos con sus reglas originales.

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import math
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
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
from robot_expedition.planetary.qlearning import PlanetaryQLearningAgent  # noqa: E402
from robot_expedition.planetary.planner import PlanetaryMissionPlanner  # noqa: E402
from pipelines.policies import CampaignNumpyPPOPolicy  # noqa: E402

API_VERSION = "1.1.0"
MAX_SEEDS = 20                 # la web corta las solicitudes a los 12 s
MAX_SESSIONS = 50
MAX_BATCH_STEPS = 40         # cada petición cruza la red (~0,45 s en Render): se agrupan pasos
SESSION_TTL_S = 15 * 60
STARTED = time.time()
DATA_DIR = Path(os.environ.get("EXPEDITION_DATA_DIR", ROOT))
EPISODE_DB = Path(os.environ.get("EXPEDITION_EPISODE_DB", DATA_DIR / "logs" / "planetary_episodes.sqlite"))
PLANETARY_Q_CHECKPOINT = Path(os.environ.get(
    "PLANETARY_Q_CHECKPOINT", DATA_DIR / "models" / "planetary_qlearning.npz"
))
PLANETARY_Q_CHECKPOINT_DIR = Path(os.environ.get(
    "PLANETARY_Q_CHECKPOINT_DIR", PLANETARY_Q_CHECKPOINT.parent
))
PLANETARY_Q_SELECTION = PLANETARY_Q_CHECKPOINT_DIR / "planetary_qlearning_active.json"
q_checkpoint_lock = threading.Lock()
QLEARNING_NO_PROGRESS_LIMIT = 120
# Checkpoint base entrenado offline (scripts/train_planetary_qlearning.py). Sin disco
# persistente, Render borra models/ en cada redeploy o reposo; al arrancar se descarga
# esta tabla solo si el checkpoint activo no existe, asi nunca se pisa lo aprendido.
# Una cadena vacia desactiva la descarga.
PLANETARY_Q_SEED_URL = os.environ.get(
    "PLANETARY_Q_SEED_URL",
    "https://github.com/nataliariosmoreno11-dev/robot_expedition/releases/download/"
    "qlearning-seed-v2/planetary_qlearning.npz",
).strip()


def _retire_incompatible_checkpoint(path: Path) -> Path | None:
    """Renombra (sin borrar) un checkpoint de otra versión del estado Q para no perderlo."""
    try:
        PlanetaryQLearningAgent().load(path)
        return None
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        backup = path.with_name(f"{path.stem}-estado-anterior.npz")
        suffix = 1
        while backup.exists():
            suffix += 1
            backup = path.with_name(f"{path.stem}-estado-anterior-{suffix}.npz")
        path.replace(backup)
        print(f"[qlearning] checkpoint incompatible conservado como {backup.name}", flush=True)
        return backup


def _seed_q_checkpoint() -> str:
    if PLANETARY_Q_CHECKPOINT.exists():
        # Un checkpoint de una versión anterior del estado Q se conserva renombrado
        # y se reemplaza por el base actual; uno compatible nunca se toca.
        _retire_incompatible_checkpoint(PLANETARY_Q_CHECKPOINT)
    if not PLANETARY_Q_SEED_URL or PLANETARY_Q_CHECKPOINT.exists():
        return "existing" if PLANETARY_Q_CHECKPOINT.exists() else "disabled"
    temporary = PLANETARY_Q_CHECKPOINT.with_name(PLANETARY_Q_CHECKPOINT.name + ".seed.tmp")
    try:
        PLANETARY_Q_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PLANETARY_Q_SEED_URL, timeout=30) as response, temporary.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        PlanetaryQLearningAgent().load(temporary)          # valida el formato antes de activarlo
        with q_checkpoint_lock:
            if PLANETARY_Q_CHECKPOINT.exists():
                return "existing"
            temporary.replace(PLANETARY_Q_CHECKPOINT)
        return "seeded"
    except Exception as exc:  # la API debe arrancar aunque GitHub no responda
        print(f"[qlearning] no se pudo descargar el checkpoint base: {type(exc).__name__}: {exc}", flush=True)
        return "failed"
    finally:
        temporary.unlink(missing_ok=True)


PLANETARY_Q_SEED_STATUS = _seed_q_checkpoint()

# Hitos PPO aprobados en la campana fija de Tomas. Se publican como evidencia
# de entrenamiento y no reemplazan el Q-learning interactivo de esta rama.
PPO_CAMPAIGN_MODELS = (
    "f7c104a304f25b458721258b540a337e",
    "d0ecdae6689f527b835c1e473e893d0c",
    "d0c438b791c156688c71cf58c69fceb5",
    "bdbc0d04fffe5c37a5a5ac5a05013889",
    "646fbc70700f5ab8b2363524540fddca",
    "156aaf20543e523eb1fa4594547d956e",
    "6109346a105e5f3e82134dc9cafbbcb9",
    "4bcfd8f641df50b9b5460d76b8c56482",
)


def _json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _campaign_evidence(model_dir: Path, level: int) -> dict | None:
    evidence_files = sorted((model_dir / "evidence").glob("*.json"))
    preferred = [path for path in evidence_files if f"-{level}" in path.stem]
    for path in reversed(preferred or evidence_files):
        payload = _json_file(path)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return {"file": path.name, "metrics": metrics,
                    "cases": len(payload.get("cases", []))}
    return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _checkpoint_candidates() -> list[Path]:
    PLANETARY_Q_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = set(PLANETARY_Q_CHECKPOINT_DIR.glob("*.npz"))
    if PLANETARY_Q_CHECKPOINT.exists():
        candidates.add(PLANETARY_Q_CHECKPOINT)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _active_checkpoint() -> Path:
    try:
        selected = json.loads(PLANETARY_Q_SELECTION.read_text(encoding="utf-8"))["name"]
        candidate = PLANETARY_Q_CHECKPOINT_DIR / Path(selected).name
        if candidate.exists() and candidate.suffix == ".npz":
            return candidate
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return PLANETARY_Q_CHECKPOINT


def _checkpoint_info(path: Path, *, active: Path | None = None) -> dict:
    agent = PlanetaryQLearningAgent()
    agent.load(path)
    stat = path.stat()
    return {
        "name": path.name,
        "path": _relative_path(path),
        "active": path.resolve() == (active or _active_checkpoint()).resolve(),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        **agent.snapshot(),
    }


store = Store(ROOT)
app = FastAPI(title="Robot Expedition API", version=API_VERSION,
              description="Simulación 3D (MuJoCo) y evaluación de modelos de Robot Expedition.")

from api.training import router as training_router, checkpoint_for
app.include_router(training_router)

DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:4173,https://web-five-gules-83.vercel.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
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
class PlanetaryRulePolicy(PlanetaryMissionPlanner):
    """Guía planificadora: mapa de misión estático + cámara; nunca ve las muestras de antemano.

    Reemplaza a la política reactiva anterior, que exploraba con arcos de 270°
    contra la pared y no completaba los niveles 2-8 (docs/auditoria-qlearning-planetario.md).
    """


class PlanetaryQLearningPolicy:
    """Q-learning persistente sobre la sugerencia del guía planificador.

    El estado incluye la acción sugerida por el guía, de modo que la tabla aprende
    en qué contexto sensorial seguirla o corregirla. Como la sugerencia del paso
    siguiente solo existe al consultarlo, cada actualización Q se completa un paso
    después (o al terminar el episodio, con valor futuro 0).
    """

    def __init__(self, seed: int, checkpoint: Path | None = None):
        self.guide = PlanetaryRulePolicy()
        self.checkpoint = Path(checkpoint or _active_checkpoint())
        self.agent = PlanetaryQLearningAgent(seed=seed)
        with q_checkpoint_lock:
            if self.checkpoint.exists() and _retire_incompatible_checkpoint(self.checkpoint):
                self.checkpoint = PLANETARY_Q_CHECKPOINT
            self.agent.load(self.checkpoint)
        self.last_state = None
        self.last_action = None
        self.pending = None
        self.shielded = 0

    def bind(self, env) -> "PlanetaryQLearningPolicy":
        self.guide.bind(env)
        return self

    def describe(self) -> dict:
        return {"algorithm": "qlearning", "guide": self.guide.name, "checkpoint": self.checkpoint.name}

    def __call__(self, observation) -> int:
        guide_action = self.guide(observation)
        state, action = self.agent.choose(observation, guide_action=guide_action)
        if self.guide.caution and action != guide_action:
            # Escudo de seguridad: junto a la grieta o un cráter no se explora; una
            # desviación ahí termina la misión. La tabla registra la acción ejecutada.
            action = guide_action
            self.shielded += 1
        if self.pending is not None:
            previous_state, previous_action, reward = self.pending
            self.agent.update(previous_state, previous_action, reward, state, False)
            self.pending = None
        self.last_state, self.last_action = state, action
        return action

    def learn(self, reward: float, next_observation, done: bool) -> None:
        if self.last_state is None or self.last_action is None:
            return
        if not done:
            self.pending = (self.last_state, self.last_action, float(reward))
            return
        self.agent.update(self.last_state, self.last_action, float(reward), None, True)
        self.agent.finish_episode()
        with q_checkpoint_lock:
            self.agent.save(self.checkpoint)

    def snapshot(self) -> dict:
        return {**self.agent.snapshot(), "shielded_decisions": self.shielded}


class Session:
    def __init__(self, level_id: str, policy, policy_id: str, seed: int,
                 environment_kind: str = "beacon", difficulty_level: int = 1):
        if environment_kind == "planetary":
            qlearning = policy_id == "planetary-qlearning"
            trained_campaign = policy_id.removeprefix("trained:") in PPO_CAMPAIGN_MODELS
            self.env = PlanetaryExplorationEnv(
                difficulty_level=difficulty_level,
                history_steps=4 if policy is not None else 1,
                curriculum=False,
                episode_db=EPISODE_DB,
                record_episodes=True,
                run_id="api-manual" if policy is None else policy_id,
                policy_version=policy_id,
                mode="manual" if policy is None else ("entrenamiento" if policy_id == "planetary-qlearning" else "evaluación"),
                update_curriculum=False,
                # Volver a la base cruza terreno ya explorado: con 35 pasos el regreso
                # se cortaba como "sin progreso". 120 sigue cortando ciclos reales.
                training_no_progress_limit=QLEARNING_NO_PROGRESS_LIMIT if qlearning else 0,
                # El episodio solo tiene éxito cuando la misión real termina.
                # La recompensa acumulada nunca sustituye esa condición. Sin piso de
                # recompensa: la batería ya limita el episodio y -20 cortaba misiones
                # válidas de ~300 pasos solo por la penalización por paso.
                training_reward_limit=0.0,
                training_reward_floor=0.0,
                training_step_penalty=0.04 if qlearning else 0.01,
                training_collision_penalty=4.0 if qlearning else 0.6,
                training_stuck_penalty=1.0 if qlearning else 0.12,
                training_no_progress_penalty=12.0 if qlearning else 2.0,
                campaign_training_compatibility=trained_campaign,
            )
            self.action_names = [item[0] for item in PLANETARY_ACTIONS]
        else:
            self.env = BeaconExpeditionEnv(level=level_id)
            self.action_names = [item[0] for item in ACTIONS]
        self.policy, self.policy_id, self.level_id = policy, policy_id, level_id
        self.environment_kind = environment_kind
        self.obs, _ = self.env.reset(seed=seed)
        # Se conserva el límite de pasos de cada escenario (650-1700): la batería es el
        # límite real. Los recortes anteriores (50/100 pasos) hacían imposibles los
        # niveles 2-8, que necesitan 90-450 pasos incluso con una ruta eficiente.
        if environment_kind == "planetary" and hasattr(policy, "bind"):
            policy.bind(self.env)       # mapa de misión estático y pose de aterrizaje
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
    previous_session_id: str | None = None


class StepRequest(BaseModel):
    session_id: str
    action: int | None = Field(default=None, ge=0, le=len(PLANETARY_ACTIONS) - 1,
                               description="acción manual; vacía = decide la política de la sesión")
    steps: int = Field(default=1, ge=1, le=MAX_BATCH_STEPS,
                       description="pasos de la política en una sola petición (solo sin action); "
                                   "se detiene antes si el episodio termina")


@app.post("/sim/reset")
def sim_reset(req: ResetRequest):
    if req.environment == "planetary" and req.policy.startswith("trained:"):
        run_id = req.policy.removeprefix("trained:")
        if run_id in PPO_CAMPAIGN_MODELS:
            adapter = ROOT / "catalog" / "campaign_adapters" / f"{run_id}.npz"
            if not adapter.is_file():
                fail(503, f"no está desplegado el adaptador PPO de {run_id}")
            policy = CampaignNumpyPPOPolicy(adapter)
        else:
            policy = store.policy(run_id)
    elif req.policy == "manual":
        policy = None
    elif req.environment == "planetary" and req.policy == "planetary-rule":
        policy = PlanetaryRulePolicy()
    elif req.environment == "planetary" and req.policy == "planetary-qlearning":
        policy = PlanetaryQLearningPolicy(req.seed)
    elif req.environment == "planetary":
        fail(422, "política planetaria desconocida")
    else:
        policy = policy_for(req.policy)
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
        if req.previous_session_id:
            previous=sessions.pop(req.previous_session_id,None)
            if previous: previous.env.close()
        _prune_sessions()
        sessions[sid] = session
    env = session.env
    return {"session_id": sid, "mode": "manual" if policy is None else "política", "policy": req.policy,
            "level_id": req.level_id, "seed": req.seed,
            "environment": req.environment,
            "environment_version": (PLANETARY_ENVIRONMENT_VERSION if req.environment == "planetary"
                                    else ENVIRONMENT_VERSION),
            "map": env.map_spec(), "frame": frame(env, -1, 0.0, False), "actions": session.action_names,
            "max_steps": env.max_steps,
            "policy_model": policy.describe() if hasattr(policy, "describe") else None,
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
    batch = 1 if req.action is not None else req.steps
    frames, events = [], []
    for _ in range(batch):
        action = req.action if req.action is not None else int(session.policy(session.obs))
        if not session.env.action_space.contains(action):
            fail(422, f"acción inválida para {session.environment_kind}: {action}")
        session.obs, reward, terminated, truncated, info = session.env.step(action)
        session.total += reward
        session.done = terminated or truncated
        if hasattr(session.policy, "learn"):
            session.policy.learn(float(reward), session.obs, session.done)
        frames.append(frame(session.env, action, reward, info["collision"]))
        if info["event"]:
            events.append({"step": session.env.steps, "event": info["event"]})
        if session.done:
            break
    session.touched = time.time()
    training = session.policy.snapshot() if hasattr(session.policy, "snapshot") else None
    # Los campos de nivel superior describen el último paso (compatibles con steps=1);
    # `frames` trae todos los cuadros del lote, en orden, y `events` sus eventos.
    return {"session_id": req.session_id, "frame": frames[-1], "frames": frames, "events": events,
            "action": action, "action_name": session.action_names[action], "reward": round(float(reward), 4),
            "return": round(session.total, 4), "terminated": terminated, "truncated": truncated,
            "end": info["end"] or "running", "event": info["event"], "stage": info["stage"],
            "battery": round(info["battery"], 2), "collisions": info["collisions"],
            "success": info["success"], "steps": session.env.steps,
            "max_steps": session.env.max_steps,
            "target_type": info.get("target_type"),
            "requested_samples": info.get("requested_samples"),
            "samples_collected": info.get("samples_collected"),
            "reward_limit": session.env.training_reward_limit if session.environment_kind == "planetary" else None,
            "reward_floor": session.env.training_reward_floor if session.environment_kind == "planetary" else None,
            "storm_intensity": info.get("storm_intensity", 0.0),
            "inventory": info.get("inventory"), "training": training,
            "decision": getattr(session.policy, "last_decision", None)}


# ------------------------------------------------------- historial planetario
def _with_episode_store(callback):
    store = EpisodeStore(EPISODE_DB)
    try:
        return callback(store)
    finally:
        store.close()


@app.get("/training/qlearning")
def qlearning_training_status():
    checkpoint_path = _active_checkpoint()
    agent = PlanetaryQLearningAgent()
    compatible = True
    with q_checkpoint_lock:
        try:
            agent.load(checkpoint_path)
        except (ValueError, KeyError, json.JSONDecodeError):
            compatible = False      # el próximo episodio lo conserva renombrado y empieza de cero
    return {
        **agent.snapshot(),
        "checkpoint": _relative_path(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_compatible": compatible,
        "guide": PlanetaryRulePolicy.name,
        "persistent_data_dir": str(DATA_DIR),
        "seed": PLANETARY_Q_SEED_STATUS,
    }


@app.get("/training/ppo/campaign")
def ppo_campaign_catalog():
    """Ocho niveles fijos y sus checkpoints PPO aprobados, listos para la defensa."""
    levels = []
    for level, run_id in enumerate(PPO_CAMPAIGN_MODELS, start=1):
        scenario_path = ROOT / "configs" / "campaign" / f"scenario_{level:02d}.json"
        model_dir = ROOT / "models" / run_id
        if not scenario_path.is_file() or not (model_dir / "approved-version.json").is_file():
            continue
        scenario = _json_file(scenario_path)
        approved = _json_file(model_dir / "approved-version.json")
        evidence = _campaign_evidence(model_dir, level)
        levels.append({
            "level": level,
            "scenario": scenario,
            "model": {
                "run_id": run_id,
                "algorithm": "PPO",
                "total_timesteps": approved.get("total_timesteps", 0),
                "completed_scenarios": approved.get("completed_scenarios", []),
                "weights_sha256": approved.get("weights_sha256"),
                "checkpoint_complete": all((model_dir / name).is_file() for name in (
                    "ppo_model.zip", "vecnormalize.pkl", "optimizer.pt", "checkpoint_state.json"
                )),
            },
            "evidence": evidence,
        })
    return {
        "schema": "ppo-fixed-campaign/v1",
        "algorithm": "PPO",
        "levels": levels,
        "latest_run_id": PPO_CAMPAIGN_MODELS[-1],
        "latest_total_timesteps": levels[-1]["model"]["total_timesteps"] if levels else 0,
        "training_available": os.environ.get("ENABLE_LOCAL_TRAINING") == "1",
        "deployment_mode": "inference-and-evidence",
    }


@app.get("/training/ppo/campaign/{level}/replay")
def ppo_campaign_replay(level: int):
    """Caso real de evaluacion que permite inspeccionar la ruta aprendida."""
    if level < 1 or level > len(PPO_CAMPAIGN_MODELS):
        fail(404, "nivel PPO desconocido")
    model_dir = ROOT / "models" / PPO_CAMPAIGN_MODELS[level - 1]
    evidence_files = sorted((model_dir / "evidence").glob("*.json"))
    preferred = [path for path in evidence_files if f"-{level}" in path.stem]
    for path in reversed(preferred or evidence_files):
        payload = _json_file(path)
        cases = payload.get("cases", [])
        matching = [case for case in cases if int(case.get("difficulty_level", 0)) == level]
        if matching:
            return {"level": level, "run_id": PPO_CAMPAIGN_MODELS[level - 1],
                    "source": path.name, "case": matching[0], "metrics": payload.get("metrics", {})}
    fail(404, "el checkpoint no contiene un caso reproducible para este nivel")


@app.get("/training/qlearning/checkpoints")
def qlearning_checkpoints():
    """Lista únicamente checkpoints compatibles con el agente planetario."""
    active = _active_checkpoint()
    compatible, incompatible = [], []
    with q_checkpoint_lock:
        for path in _checkpoint_candidates():
            try:
                compatible.append(_checkpoint_info(path, active=active))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                incompatible.append({"name": path.name, "reason": str(exc)})
    return {
        "active": active.name,
        "directory": str(PLANETARY_Q_CHECKPOINT_DIR),
        "persistent": DATA_DIR.resolve() != ROOT.resolve(),
        "checkpoints": compatible,
        "incompatible": incompatible,
    }


class ActivateCheckpointRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@app.post("/training/qlearning/checkpoints/activate")
def activate_qlearning_checkpoint(req: ActivateCheckpointRequest):
    name = Path(req.name).name
    if name != req.name or not re.fullmatch(r"[A-Za-z0-9_.-]+\.npz", name):
        fail(422, "nombre de checkpoint inválido")
    candidate = PLANETARY_Q_CHECKPOINT_DIR / name
    if not candidate.is_file():
        fail(404, f"no existe el checkpoint {name}")
    with q_checkpoint_lock:
        try:
            info = _checkpoint_info(candidate, active=candidate)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            fail(409, f"checkpoint incompatible: {exc}")
        PLANETARY_Q_SELECTION.write_text(json.dumps({"name": name}), encoding="utf-8")
    return {**info, "message": "checkpoint activado; el próximo episodio continuará desde este estado"}


class SnapshotCheckpointRequest(BaseModel):
    name: str = Field(default="", max_length=80)


@app.post("/training/qlearning/checkpoints/snapshot")
def snapshot_qlearning_checkpoint(req: SnapshotCheckpointRequest):
    source = _active_checkpoint()
    if not source.exists():
        fail(409, "todavía no existe un checkpoint para guardar")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", req.name.strip()).strip("-")
    if not stem:
        stem = datetime.now(timezone.utc).strftime("checkpoint-%Y%m%d-%H%M%S")
    destination = PLANETARY_Q_CHECKPOINT_DIR / f"{stem}.npz"
    if destination.exists():
        fail(409, f"ya existe {destination.name}; usa otro nombre")
    with q_checkpoint_lock:
        shutil.copy2(source, destination)
        return _checkpoint_info(destination)


# ------------------------------------------------------- historial planetario
def _with_episode_store(callback):
    store = EpisodeStore(EPISODE_DB)
    try:
        return callback(store)
    finally:
        store.close()



@app.get("/episodes")
def episodes(limit: int = 100, offset: int = 0, run_id: str | None = None, scenario_id: str | None = None, mode: str | None = None):
    return {
        "schema_version": 2,
        "database": EPISODE_DB.name,
        "storage": _with_episode_store(lambda episode_store: episode_store.storage_status()),
        "episodes": _with_episode_store(
            lambda episode_store: episode_store.list_episodes(limit=limit, offset=offset, run_id=run_id, scenario_id=scenario_id, mode=mode)
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


@app.get("/scenarios")
def campaign_catalog():
    from robot_expedition.planetary.campaign import SCENARIO_IDS, load_scenario, catalog_hash
    from robot_expedition.planetary.reference import campaign_reference
    return {"catalog_hash":catalog_hash(),"scenarios":[{
        **load_scenario(sid).to_dict(include_heightfield=False),
        "analysis":{k:v for k,v in campaign_reference(sid).items() if k!="reference_path"}
    } for sid in SCENARIO_IDS]}


@app.get("/scenarios/{scenario_id}/analysis")
def scenario_analysis(scenario_id: str):
    from robot_expedition.planetary.reference import campaign_reference
    try: return campaign_reference(scenario_id)
    except ValueError as exc: fail(422,str(exc))


@app.get("/scenarios/{scenario_id}/history")
def scenario_history(scenario_id: str, limit: int=100, offset: int=0, mode: str | None=None):
    return _with_episode_store(lambda store: store.get_scenario_history(scenario_id,limit=limit,offset=offset,mode=mode))


@app.get("/scenarios/{scenario_id}/progress")
def scenario_progress(scenario_id: str, mode: str="entrenamiento"):
    return _with_episode_store(lambda store: store.scenario_report(scenario_id,mode=mode))


@app.get("/campaign/progress")
def campaign_progress():
    # El run mas avanzado: mas niveles aprobados y, a igualdad, mas pasos. No se usa la fecha
    # del archivo: en un despliegue todos los archivos se escriben a la vez y el orden es arbitrario.
    def progress(item):
        evaluations = item[1].get("evaluations") or {}
        passed = sum(bool(v.get("passed")) for v in evaluations.values())
        steps = max([int(v.get("training_step") or 0) for v in evaluations.values()] or [0])
        return passed, steps, item[0]

    states = []
    for path in (ROOT/"models").glob("*/curriculum_state.json"):
        state = json.loads(path.read_text())
        if state.get("schema") == "fixed-campaign-curriculum/v1":
            states.append((path.parent.name, state))
    if states:
        run_id, state = max(states, key=progress)
        return {"run_id": run_id, **state}
    from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum
    return {"run_id":None,**CampaignCurriculum().snapshot()}
