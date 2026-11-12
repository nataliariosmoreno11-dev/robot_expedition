"""Reproducible, one-run-per-seed training for Robot Expedition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:  # Las utilidades de config/manifiesto siguen disponibles sin la pila pesada.
    import mlflow
    import torch
    from mlflow.tracking import MlflowClient
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ModuleNotFoundError:  # pragma: no cover - la ruta se comprueba antes de entrenar
    mlflow = torch = MlflowClient = PPO = Monitor = DummyVecEnv = VecNormalize = None
    set_random_seed = None

    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.model = None
            self.locals = {}

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_expedition.envs.beacon_expedition import (  # noqa: E402
    ENVIRONMENT_VERSION, BeaconExpeditionEnv, environment_hash, load_level_config, load_scenario_set,
)
from robot_expedition.envs.planetary_exploration import (  # noqa: E402
    OBSERVATION_SIZE,
    PLANETARY_ENVIRONMENT_VERSION,
    PlanetaryExplorationEnv,
    planetary_environment_hash,
)
from pipelines.evaluate import evaluate  # noqa: E402
from pipelines.planetary_evaluate import evaluate_planetary  # noqa: E402
from pipelines.policies import PPOPolicy, QTablePolicy  # noqa: E402

N_ENVS = 4


def _require_training_stack() -> None:
    if any(item is None for item in (mlflow, torch, PPO, Monitor, DummyVecEnv, VecNormalize)):
        raise RuntimeError(
            "Falta la pila de entrenamiento. Usa Python 3.12 e instala requirements.txt "
            "antes de ejecutar pipelines/train.py."
        )


def evaluation_suite(level_id: str, configured: str | None = None) -> str:
    """Suite fija del paso 2 para el nivel (configs/suites)."""
    return configured or f"suite-{level_id}-v1"


def is_planetary(config: dict[str, Any]) -> bool:
    return config.get("environment", {}).get("kind") == "planetary"


def config_environment_version(config: dict[str, Any]) -> str:
    return PLANETARY_ENVIRONMENT_VERSION if is_planetary(config) else ENVIRONMENT_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def sha256_json(data: Any) -> str:
    return hashlib.sha256(_json_text(data).encode("utf-8")).hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(data), encoding="utf-8")
    temporary.replace(path)


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"La configuración debe ser un mapa YAML: {path}")
    required = ("experiment_id", "hypothesis", "level_id", "algorithm", "budget_timesteps", "seeds")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Faltan campos de configuración: {', '.join(missing)}")
    if config["algorithm"] not in ("ppo", "qlearning"):
        raise ValueError("algorithm debe ser 'ppo' o 'qlearning'")
    if config["level_id"] not in ("n1", "n2", "n3"):
        raise ValueError("level_id debe ser n1, n2 o n3 (configs/levels)")
    load_level_config(config["level_id"])
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("seeds debe ser una lista no vacía")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in config["seeds"]):
        raise ValueError("cada elemento de seeds debe ser un entero")
    if int(config["budget_timesteps"]) <= 0:
        raise ValueError("budget_timesteps debe ser positivo")
    config.setdefault("hyperparams", {})
    if not isinstance(config["hyperparams"], dict):
        raise TypeError("hyperparams debe ser un mapa")
    config.setdefault("normalize_obs", True)
    config.setdefault("parent_run_id", None)
    config.setdefault("environment", {})
    if not isinstance(config["environment"], dict):
        raise TypeError("environment debe ser un mapa")
    if config["environment"].get("kind") == "planetary":
        if config["algorithm"] != "ppo":
            raise ValueError("el ambiente planetario usa PPO; Q-learning conserva el contrato Box(20)/Discrete(5)")
        level = int(config["environment"].get("difficulty_level", 1))
        if not 1 <= level <= 8:
            raise ValueError("environment.difficulty_level debe estar entre 1 y 8")
    else:
        # El ruido de rayos, rocas, dunas y bloqueo salen del nivel; environment
        # solo puede sobrescribirlos de forma explícita.
        config["environment"] = {
            "max_steps": 400,
            "rock_jitter": 0.0,
            **config["environment"],
        }
        if config["environment"].get("scenario_set") is not None:
            load_scenario_set(config["environment"]["scenario_set"])
    return config


def build_env(config: dict[str, Any]):
    options = dict(config["environment"])
    kind = options.pop("kind", "beacon")
    if kind == "planetary":
        return PlanetaryExplorationEnv(**options)
    return BeaconExpeditionEnv(level=config["level_id"], **options)


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def configure_tracking() -> tuple[str, str]:
    _require_training_stack()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{ROOT / 'mlflow.db'}",
    )
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")
    if registry_uri is None:
        registry_uri = (
            f"sqlite:///{ROOT / 'mlflow.db'}"
            if tracking_uri.startswith("file:")
            else tracking_uri
        )
    mlflow.set_registry_uri(registry_uri)
    if tracking_uri.startswith("sqlite:"):
        client = MlflowClient()
        if client.get_experiment_by_name("robot-expedition") is None:
            client.create_experiment(
                "robot-expedition",
                artifact_location=(ROOT / "mlruns").as_uri(),
            )
    mlflow.set_experiment("robot-expedition")
    return tracking_uri, registry_uri


def _flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_params(item, name))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(item, sort_keys=True)
        elif item is None:
            flattened[name] = "null"
        else:
            flattened[name] = str(item)
    return flattened


def _make_env(config: dict[str, Any], seed: int | None = None) -> Monitor:
    _require_training_stack()
    env = build_env(config)
    if seed is not None:
        env.reset(seed=seed)
    info_keywords = ("success", "samples_collected", "difficulty_level") if is_planetary(config) else ("success",)
    return Monitor(env, info_keywords=info_keywords)


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, step_offset: int = 0):
        super().__init__()
        self.step_offset = step_offset

    def _on_step(self) -> bool:
        step = max(1, int(self.model.num_timesteps) - self.step_offset)
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                mlflow.log_metric("training/reward", float(episode["r"]), step=step)
                mlflow.log_metric(
                    "training/success_rate",
                    float(bool(episode.get("success", False))),
                    step=step,
                )
                if "samples_collected" in episode:
                    mlflow.log_metric(
                        "training/samples_collected", float(episode["samples_collected"]), step=step
                    )
                if "difficulty_level" in episode:
                    mlflow.log_metric(
                        "training/difficulty_level", float(episode["difficulty_level"]), step=step
                    )
        return True


class _LivePPOPolicy:
    """Vista determinista de la política actual con la normalización congelada."""

    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer

    def __call__(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        if self.normalizer is not None:
            observation = self.normalizer.normalize_obs(observation)
        action, _ = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


class PeriodicPlanetaryEvaluationCallback(BaseCallback):
    """Evalúa snapshots en un holdout fijo sin modificar entorno ni currículo de training."""

    def __init__(
        self,
        config: dict[str, Any],
        model_dir: Path,
        *,
        frequency: int,
        episodes: int,
        seed_start: int,
        step_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.model_dir = model_dir
        self.frequency = max(1, int(frequency))
        self.episodes = max(1, int(episodes))
        self.seed_start = int(seed_start)
        self.next_step = int(step_offset) + self.frequency

    def _on_step(self) -> bool:
        step = int(self.model.num_timesteps)
        if step < self.next_step:
            return True
        training_env = self.model.get_env()
        policy = _LivePPOPolicy(self.model, training_env)
        result = evaluate_planetary(
            policy,
            self.config["environment"],
            episodes=self.episodes,
            seed_start=self.seed_start,
            max_level=int(self.config.get("evaluation_max_level", 8)),
        )
        result["suite_id"] = "planetary-periodic-holdout-v1"
        result["training_step"] = step
        save_json(self.model_dir / f"periodic-evaluation-{step}.json", result)
        for key in (
            "success_rate",
            "return_mean",
            "collisions_per_episode",
            "samples_collected_mean",
            "mission_completion",
        ):
            value = result["metrics"].get(key)
            if value is not None:
                mlflow.log_metric(f"periodic_evaluation/{key}", float(value), step=step)
        while self.next_step <= step:
            self.next_step += self.frequency
        return True


def _train_qlearning(
    config: dict[str, Any],
    seed: int,
    budget: int,
    model_dir: Path,
    parent_run_id: str | None,
) -> tuple[Path, int]:
    from robot_expedition.qlearning import QLearningAgent, discretize

    env = build_env(config)
    hyperparams = config["hyperparams"]
    agent = QLearningAgent(
        env.action_space.n,
        alpha=float(hyperparams.get("alpha", 0.2)),
        gamma=float(hyperparams.get("gamma", 0.98)),
        epsilon=1.0,
        seed=seed,
    )
    step_offset = 0
    if parent_run_id:
        checkpoint = _download_checkpoint(parent_run_id)
        agent.load(checkpoint / "qlearning.npz")
        step_offset = int(json.loads((checkpoint / "checkpoint_state.json").read_text())["total_timesteps"])
        if step_offset != agent.steps:
            raise ValueError("El contador de pasos del checkpoint Q-learning no coincide")

    episodes = 0
    while agent.steps - step_offset < budget:
        obs, _ = env.reset(seed=seed * 1_000_000 + episodes)
        state = discretize(obs)
        episode_reward = 0.0
        success = False
        while agent.steps - step_offset < budget:
            action = agent.choose(state, explore=True)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize(obs)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            success = bool(info["success"])
            agent.epsilon = max(
                float(hyperparams.get("epsilon_min", 0.05)),
                1.0
                - (1.0 - float(hyperparams.get("epsilon_min", 0.05)))
                * (agent.steps - step_offset)
                / max(1, int(0.7 * budget)),
            )
            if terminated or truncated:
                break
        episodes += 1
        local_step = agent.steps - step_offset
        mlflow.log_metric("training/reward", float(episode_reward), step=local_step)
        mlflow.log_metric("training/success_rate", float(success), step=local_step)
    model_path = model_dir / "qlearning.npz"
    agent.save(model_path)
    save_json(model_dir / "checkpoint_state.json", {"total_timesteps": agent.steps})
    return model_path, agent.steps - step_offset


def _download_checkpoint(parent_run_id: str) -> Path:
    checkpoint_dir = Path(
        mlflow.artifacts.download_artifacts(run_id=parent_run_id, artifact_path="checkpoint")
    )
    state_path = checkpoint_dir / "checkpoint_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"El run padre {parent_run_id} no tiene checkpoint_state.json")
    return checkpoint_dir


def _restore_curriculum_checkpoint(parent_checkpoint: Path, config: dict[str, Any]) -> Path | None:
    if not is_planetary(config):
        return None
    saved = parent_checkpoint / "curriculum_state.json"
    target_value = config["environment"].get("curriculum_state")
    if not saved.is_file() or not target_value:
        return None
    target = Path(target_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved, target)
    return target


def _train_ppo(
    config: dict[str, Any],
    seed: int,
    budget: int,
    model_dir: Path,
    parent_run_id: str | None,
    n_steps_override: int | None,
    batch_size_override: int | None,
) -> tuple[Path, int, Path]:
    _require_training_stack()
    if not config["normalize_obs"]:
        raise ValueError("PPO requiere normalize_obs: true")
    set_random_seed(seed)
    parent_checkpoint = _download_checkpoint(parent_run_id) if parent_run_id else None
    if parent_checkpoint:
        _restore_curriculum_checkpoint(parent_checkpoint, config)
    factories = [
        lambda cfg=config: _make_env(cfg)
        for _ in range(N_ENVS)
    ]
    base_env = DummyVecEnv(factories)
    base_env.seed(seed)

    parent_steps = 0
    if parent_checkpoint:
        parent_steps = int(
            json.loads((parent_checkpoint / "checkpoint_state.json").read_text())["total_timesteps"]
        )
        vec_env = VecNormalize.load(parent_checkpoint / "vecnormalize.pkl", base_env)
        vec_env.training = True
        vec_env.norm_reward = False
        model = PPO.load(parent_checkpoint / "ppo_model.zip", env=vec_env, device="auto")
        optimizer_state = torch.load(
            parent_checkpoint / "optimizer.pt",
            map_location=model.device,
            weights_only=True,
        )
        model.policy.optimizer.load_state_dict(optimizer_state)
        if model.num_timesteps != parent_steps:
            raise ValueError("El contador de pasos PPO no coincide con checkpoint_state.json")
    else:
        vec_env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        hyperparams = dict(config["hyperparams"])
        hyperparams["n_steps"] = n_steps_override or int(hyperparams.get("n_steps", 2048))
        hyperparams["batch_size"] = batch_size_override or int(hyperparams.get("batch_size", 64))
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            verbose=0,
            **hyperparams,
        )

    callbacks = [TrainingMetricsCallback(parent_steps)]
    periodic_frequency = int(config.get("periodic_evaluation_freq", 0))
    if is_planetary(config) and periodic_frequency > 0:
        callbacks.append(
            PeriodicPlanetaryEvaluationCallback(
                config,
                model_dir,
                frequency=periodic_frequency,
                episodes=int(config.get("periodic_evaluation_episodes", 8)),
                seed_start=int(config.get("periodic_evaluation_seed_start", 910_000)),
                step_offset=parent_steps,
            )
        )
    model.learn(
        total_timesteps=budget,
        callback=callbacks,
        reset_num_timesteps=parent_run_id is None,
        progress_bar=False,
    )
    total_steps = int(model.num_timesteps)
    steps_trained = total_steps - parent_steps
    model_path = model_dir / "ppo_model.zip"
    model.save(model_path)
    vecnormalize_path = model_dir / "vecnormalize.pkl"
    vec_env.save(vecnormalize_path)
    torch.save(model.policy.optimizer.state_dict(), model_dir / "optimizer.pt")
    save_json(model_dir / "checkpoint_state.json", {"total_timesteps": total_steps})
    return model_path, steps_trained, vecnormalize_path


def _evaluate(
    config: dict[str, Any], model_path: Path, normalization_path: Path | None, episodes: int | None
) -> dict[str, Any]:
    """Evalua el modelo congelado con la suite fija del nivel (paso 2).

    episodes limita la suite para pruebas cortas; el resultado queda marcado como parcial.
    """
    if config["algorithm"] == "qlearning":
        policy = QTablePolicy(model_path)
    else:
        policy = PPOPolicy(model_path, normalization_path)
    if is_planetary(config):
        return evaluate_planetary(
            policy,
            config["environment"],
            episodes=episodes or 24,
            max_level=int(config.get("evaluation_max_level", 8)),
        )
    suite_id = evaluation_suite(config["level_id"], config.get("evaluation_suite_id"))
    return evaluate(policy, config["level_id"], suite_id, episodes=episodes)


def _register_dvc_model(model_dir: Path, run_id: str) -> str:
    dvc_executable = shutil.which("dvc")
    if dvc_executable is None:
        executable_name = "dvc.exe" if os.name == "nt" else "dvc"
        dvc_executable = str(Path(sys.executable).with_name(executable_name))
    # Varios entrenamientos en paralelo comparten el lock del repositorio DVC: se reintenta.
    for attempt in range(30):
        result = subprocess.run(
            [
                dvc_executable,
                "add",
                "--force",
                str(model_dir.relative_to(ROOT)),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 or "lock" not in (result.stderr + result.stdout).lower():
            break
        time.sleep(1 + attempt % 5)
    if result.returncode:
        raise RuntimeError(f"dvc add falló: {result.stderr.strip() or result.stdout.strip()}")
    return f"dvc://dagshub/{model_dir.relative_to(ROOT).as_posix()}"


def _manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    source_config_hash: str,
    seed: int,
    git_commit: str,
    origin: str,
    budget: int,
    effective_budget: int,
    parent_run_id: str | None,
    status: str,
    diagnosis: str | None,
    model_uri: str | None,
    model_hash: str | None,
    normalization_uri: str | None,
    metrics: dict[str, float],
    checkpoint_uri: str | None,
) -> dict[str, Any]:
    planetary = is_planetary(config)
    env_hash = (
        planetary_environment_hash(config.get("environment"))
        if planetary
        else environment_hash(config["level_id"], config.get("environment"))
    )
    return {
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "level_id": config["level_id"],
        "algorithm": config["algorithm"],
        "environment_version": config_environment_version(config),
        "environment_kind": "planetary" if planetary else "beacon",
        "git_commit": git_commit,
        "origin": {
            "type": "experiment_config",
            "config_path": str(config_path.relative_to(ROOT))
            if config_path.is_relative_to(ROOT)
            else str(config_path),
            "repository": origin,
        },
        "config_sha256": config_hash,
        "source_config_sha256": source_config_hash,
        "model_uri": model_uri,
        "model_sha256": model_hash,
        "normalization_uri": normalization_uri,
        "environment_hash": env_hash,
        "obs_spec": (
            f"{OBSERVATION_SIZE} floats, ver docs/contratos-planetario.md"
            if planetary
            else "20 floats, ver docs/contratos.md"
        ),
        "action_spec": "Discrete(6)" if planetary else "Discrete(5)",
        "runtime_ref": f"requirements.txt + python {platform.python_version()}",
        "evaluation_suite_id": (
            "planetary-holdout-v1"
            if planetary
            else evaluation_suite(config["level_id"], config.get("evaluation_suite_id"))
        ),
        "seed": seed,
        "budget_timesteps": budget,
        "effective_budget_timesteps": effective_budget,
        "parent_run_id": parent_run_id,
        "status": status,
        "diagnosis": diagnosis,
        "checkpoint_uri": checkpoint_uri,
        "metrics": metrics,
    }


def train_seed(
    config: dict[str, Any],
    config_path: Path,
    source_config_hash: str,
    seed: int,
    budget_override: int | None = None,
    eval_episodes: int | None = None,
    n_steps_override: int | None = None,
    batch_size_override: int | None = None,
    max_episode_steps: int | None = None,
) -> str:
    run_id = uuid.uuid4().hex
    status = "failed"
    diagnosis: str | None = None
    model_hash: str | None = None
    model_uri: str | None = None
    normalization_uri: str | None = None
    checkpoint_uri: str | None = None
    effective_budget = 0
    metrics: dict[str, float] = {}
    model_dir = ROOT / "models" / run_id
    budget = int(budget_override or config["budget_timesteps"])
    resolved = {
        **config,
        "seed": seed,
        "budget_timesteps": budget,
        "eval_episodes": eval_episodes,
        "n_envs": N_ENVS if config["algorithm"] == "ppo" else 1,
        "hyperparams": dict(config["hyperparams"]),
    }
    if max_episode_steps is not None:
        resolved["environment"] = {**config["environment"], "max_steps": max_episode_steps}
        config = {**config, "environment": resolved["environment"]}
    if n_steps_override is not None:
        resolved["hyperparams"]["n_steps"] = n_steps_override
    if batch_size_override is not None:
        resolved["hyperparams"]["batch_size"] = batch_size_override
    config_hash = sha256_json(resolved)

    tags = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "environment_version": config_environment_version(config),
        "level_id": str(config["level_id"]),
        "experiment_id": str(config["experiment_id"]),
        "hypothesis": str(config["hypothesis"]),
        "seed": str(seed),
    }
    params = _flatten_params(resolved)
    run = None
    previous_handlers = {}

    def write_manifest() -> Path:
        manifest = _manifest(
            run_id=run_id,
            config=config,
            config_path=config_path,
            config_hash=config_hash,
            source_config_hash=source_config_hash,
            seed=seed,
            git_commit=tags["git_commit"],
            origin=_git_value("remote", "get-url", "origin"),
            budget=budget,
            effective_budget=effective_budget,
            parent_run_id=config.get("parent_run_id"),
            status=status,
            diagnosis=diagnosis,
            model_uri=model_uri,
            model_hash=model_hash,
            normalization_uri=normalization_uri,
            metrics=metrics,
            checkpoint_uri=checkpoint_uri,
        )
        manifest_path = ROOT / "manifests" / f"{run_id}.json"
        save_json(manifest_path, manifest)
        return manifest_path

    try:
        try:
            run = mlflow.start_run(run_name=f"{config['experiment_id']}-seed-{seed}", tags=tags)
        except Exception as exc:
            status = "failed"
            diagnosis = f"{type(exc).__name__}: {exc}; sin checkpoint disponible"
            write_manifest()
            raise
        run_id = run.info.run_id
        model_dir = ROOT / "models" / run_id
        runtime_config = config
        if is_planetary(config):
            # Cada intento persistido identifica el run que realmente produjo sus
            # acciones, aunque el YAML use un nombre descriptivo estable.
            runtime_config = {
                **config,
                "environment": {
                    **config["environment"],
                    "run_id": run_id,
                    "policy_version": run_id,
                },
            }

        def handle_interrupt(received: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"Señal {received} recibida")

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_interrupt)
        try:
            mlflow.log_params(params)
            model_dir.mkdir(parents=True, exist_ok=True)
            save_json(model_dir / "config.json", resolved)
            mlflow.log_artifact(str(model_dir / "config.json"))

            if config["algorithm"] == "qlearning":
                model_path, effective_budget = _train_qlearning(
                    config, seed, budget, model_dir, config.get("parent_run_id")
                )
                normalization_path = model_dir / "vecnormalize.pkl"
                with normalization_path.open("wb") as stream:
                    pickle.dump(None, stream)
            else:
                model_path, effective_budget, normalization_path = _train_ppo(
                    runtime_config,
                    seed,
                    budget,
                    model_dir,
                    config.get("parent_run_id"),
                    n_steps_override,
                    batch_size_override,
                )

            evaluation = _evaluate(
                runtime_config,
                model_path,
                normalization_path if config["algorithm"] == "ppo" else None,
                eval_episodes,
            )
            suite_metrics = evaluation["metrics"]
            metrics = {
                "success_rate": suite_metrics["success_rate"],
                "evaluation_reward": suite_metrics["return_mean"],
                "collisions_per_episode": suite_metrics["collisions_per_episode"],
                "steps_mean": suite_metrics["steps_mean"],
                "failures_by_cause": suite_metrics["failures_by_cause"],
                "evaluation_episodes": suite_metrics["episodes"],
                "evaluation_partial": evaluation["partial"],
            }
            save_json(model_dir / "evaluacion.json", evaluation)
            # Curva de evaluacion determinista, separada de training/* (con exploracion).
            for key in ("success_rate", "return_mean", "collisions_per_episode", "steps_mean", "latency_ms_mean"):
                if suite_metrics.get(key) is not None:
                    mlflow.log_metric(f"evaluation/{key}", float(suite_metrics[key]), step=effective_budget)
            model_hash = sha256_file(model_path)
            curriculum_value = config.get("environment", {}).get("curriculum_state")
            if is_planetary(config) and curriculum_value:
                curriculum_path = Path(curriculum_value)
                if curriculum_path.exists():
                    shutil.copy2(curriculum_path, model_dir / "curriculum_state.json")
            normalization_uri = (
                f"runs:/{run_id}/vecnormalize.pkl"
                if config["algorithm"] == "ppo"
                else None
            )
            checkpoint_uri = f"runs:/{run_id}/checkpoint"

            mlflow.log_artifact(str(model_dir / "vecnormalize.pkl"))
            mlflow.log_artifact(str(model_dir / "evaluacion.json"))
            mlflow.log_artifacts(str(model_dir), artifact_path="checkpoint"            )
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(ROOT / "pipelines" / "registry_model.py"),
                artifacts={
                    "weights": str(model_path),
                    "vecnormalize": str(normalization_path),
                    "config": str(model_dir / "config.json"),
                },
                code_paths=[str(ROOT / "robot_expedition")],
                input_example=[[0.0] * (OBSERVATION_SIZE if is_planetary(config) else 20)],
            )
            model_uri = _register_dvc_model(model_dir, run_id)
            mlflow.register_model(
                model_info.model_uri,
                f"robot-expedition-{config['experiment_id']}",
            )
            status = "completed"
        except KeyboardInterrupt as exc:
            status = "interrupted"
            diagnosis = f"{type(exc).__name__}: {exc}"
            if checkpoint_uri is None:
                diagnosis += "; sin checkpoint disponible"
            raise
        except Exception as exc:
            status = "failed"
            diagnosis = f"{type(exc).__name__}: {exc}"
            if checkpoint_uri is None:
                diagnosis += "; sin checkpoint disponible"
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            manifest_path = write_manifest()
            mlflow.set_tag("status", status)
            mlflow.log_artifact(str(manifest_path))
    finally:
        if run is not None:
            run_status = (
                "FINISHED"
                if status == "completed"
                else "KILLED"
                if status == "interrupted"
                else "FAILED"
            )
            mlflow.end_run(status=run_status)
    return run_id


def run(config_path: Path, args: argparse.Namespace) -> list[str]:
    config_path = config_path.resolve()
    config = read_config(config_path)
    source_config_hash = sha256_file(config_path)
    configure_tracking()
    selected_seeds = [args.seed] if args.seed is not None else config["seeds"]
    run_ids = []
    for seed in selected_seeds:
        run_ids.append(
            train_seed(
                config,
                config_path,
                source_config_hash,
                int(seed),
                args.budget_timesteps,
                args.eval_episodes,
                args.n_steps,
                args.batch_size,
                args.max_episode_steps,
            )
        )
    return run_ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget-timesteps", type=int)
    parser.add_argument("--eval-episodes", type=int,
                        help="limita la suite fija de evaluacion (por defecto, completa: 50 episodios)")
    parser.add_argument("--seed", type=int, help="ejecuta solo esta semilla del experimento")
    parser.add_argument("--n-steps", type=int, help="override PPO para pruebas cortas")
    parser.add_argument("--batch-size", type=int, help="override PPO para pruebas cortas")
    parser.add_argument("--max-episode-steps", type=int, help="override del límite del episodio")
    args = parser.parse_args(argv)
    if args.budget_timesteps is not None and args.budget_timesteps <= 0:
        parser.error("--budget-timesteps debe ser positivo")
    if args.eval_episodes is not None and args.eval_episodes <= 0:
        parser.error("--eval-episodes debe ser positivo")
    if args.n_steps is not None and args.n_steps <= 0:
        parser.error("--n-steps debe ser positivo")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size debe ser positivo")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps debe ser positivo")
    for run_id in run(args.config, args):
        print(f"Run {run_id}: manifests/{run_id}.json")


if __name__ == "__main__":
    main()
