"""Politicas evaluables con la interfaz comun policy(obs) -> accion.

- RandomPolicy: referencia aleatoria (semilla por caso, reproducible).
- RulePolicy: referencia de regla (gira hacia el objetivo y esquiva).
- QTablePolicy: adaptador del Q-learning tabular (archivo .npz).
- PPOPolicy: adaptador de Stable-Baselines3 PPO (+ VecNormalize opcional).

Indices de la observacion (docs/contratos.md): 0-8 rayos de -90 a +90 grados
(derecha a izquierda), 9 distancia al objetivo, 10-11 sen/cos del angulo
relativo, 12 v, 13 w, 14 bateria, 15-17 etapa, 18-19 cabeceo y alabeo.
Acciones: 0 avanzar, 1 avanzar+izq, 2 avanzar+der, 3 girar izq, 4 girar der.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

FORWARD, FORWARD_LEFT, FORWARD_RIGHT, TURN_LEFT, TURN_RIGHT = range(5)


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RandomPolicy:
    """Accion uniforme; reset(seed) la hace reproducible por caso."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(5))

    def describe(self) -> dict:
        return {"policy": self.name, "kind": "referencia aleatoria"}


class RulePolicy:
    """Regla fija: si el frente esta bloqueado, gira hacia el lado con mas
    espacio; si un costado esta muy cerca, se aparta; si no, apunta al
    objetivo. No aprende ni usa la posicion absoluta."""

    name = "rule"

    def __init__(self, front_clearance: float = 0.8, side_clearance: float = 0.6,
                 aligned: float = 0.35, sharp: float = 1.2):
        self.front_clearance, self.side_clearance = front_clearance, side_clearance
        self.aligned, self.sharp = aligned, sharp

    def __call__(self, obs) -> int:
        rays = np.asarray(obs[:9])
        angle = math.atan2(float(obs[10]), float(obs[11]))      # + = objetivo a la izquierda
        if rays[3:6].min() < self.front_clearance:
            return TURN_LEFT if rays[6:9].min() > rays[0:3].min() else TURN_RIGHT
        # Sin esta separacion lateral la regla raspa las rocas (72 choques/ep. en N1).
        if rays[0:3].min() < self.side_clearance:
            return FORWARD_LEFT
        if rays[6:9].min() < self.side_clearance:
            return FORWARD_RIGHT
        if abs(angle) < self.aligned:
            return FORWARD
        if angle > 0:
            return TURN_LEFT if angle > self.sharp else FORWARD_LEFT
        return TURN_RIGHT if -angle > self.sharp else FORWARD_RIGHT

    def describe(self) -> dict:
        return {"policy": self.name, "kind": "referencia de regla",
                "params": {"front_clearance_m": self.front_clearance, "side_clearance_m": self.side_clearance,
                           "aligned_rad": self.aligned, "sharp_rad": self.sharp}}


class QTablePolicy:
    """Q-learning tabular congelado: argmax de la tabla (primer maximo)."""

    name = "qlearning"

    def __init__(self, model_path: str | Path):
        from robot_expedition.qlearning import discretize
        self.model_path = Path(model_path)
        self._discretize = discretize
        self.q = np.load(self.model_path)["q"]

    def __call__(self, obs) -> int:
        return int(np.argmax(self.q[self._discretize(obs)]))

    def describe(self) -> dict:
        return {"policy": self.name, "kind": "modelo aprendido", "model_path": str(self.model_path),
                "model_sha256": sha256_file(self.model_path),
                "discretization": "etapa x distancia x angulo x 3 sectores de rayos (2592 estados)"}


class PPOPolicy:
    """PPO de Stable-Baselines3 en modo determinista. Si hay VecNormalize,
    aplica exactamente la misma normalizacion de observaciones del entrenamiento."""

    name = "ppo"

    def __init__(self, model_path: str | Path, vecnormalize_path: str | Path | None = None):
        from stable_baselines3 import PPO
        self.model_path = Path(model_path)
        self.vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path else None
        self.model = PPO.load(self.model_path, device="cpu")
        self.normalizer = None
        if self.vecnormalize_path:
            with open(self.vecnormalize_path, "rb") as fh:          # artefacto propio del equipo
                self.normalizer = pickle.load(fh)

    def __call__(self, obs) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if self.normalizer is not None and self.normalizer.norm_obs:
            obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def describe(self) -> dict:
        return {"policy": self.name, "kind": "modelo aprendido", "model_path": str(self.model_path),
                "model_sha256": sha256_file(self.model_path),
                "vecnormalize_path": str(self.vecnormalize_path) if self.vecnormalize_path else None,
                "vecnormalize_sha256": sha256_file(self.vecnormalize_path)}


class NumpyPPOPolicy:
    """Adaptador sin torch de un PPO MlpPolicy (20 -> 64 -> 64 -> 5, tanh) + VecNormalize.

    Reproduce PPO.predict(deterministic=True): normaliza la observacion igual que
    VecNormalize.normalize_obs y toma el argmax de los logits. Se exporta con
    export_ppo_numpy y su compatibilidad se valida con scripts/export_numpy_policies.py
    (mismos casos que el modelo original en las suites fijas).
    """

    name = "ppo"

    def __init__(self, adapter_path: str | Path):
        self.adapter_path = Path(adapter_path)
        data = np.load(self.adapter_path)
        self.layers = [(data["w0"], data["b0"]), (data["w1"], data["b1"])]
        self.action = (data["wa"], data["ba"])
        self.norm = bool(data["norm_obs"])
        self.mean, self.var = data["obs_mean"], data["obs_var"]
        self.clip, self.eps = float(data["clip_obs"]), float(data["epsilon"])
        self.source_sha256 = str(data["source_sha256"])

    def __call__(self, obs) -> int:
        x = np.asarray(obs, dtype=np.float32)
        if self.norm:
            x = np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        w, b = self.action
        return int(np.argmax(w @ x + b))

    def describe(self) -> dict:
        return {"policy": self.name, "kind": "modelo aprendido (adaptador numpy validado)",
                "adapter_path": str(self.adapter_path), "adapter_sha256": sha256_file(self.adapter_path),
                "model_sha256": self.source_sha256}


class CampaignNumpyPPOPolicy:
    """PPO planetario especializado exportado fielmente a NumPy.

    Conserva la normalización, los actores por escenario y la rama de valor.
    ``last_decision`` expone únicamente resultados reales del forward pass.
    """

    name = "ppo"

    def __init__(self, adapter_path: str | Path):
        self.adapter_path = Path(adapter_path)
        data = np.load(self.adapter_path)
        self.actor_count = int(data["actor_count"])
        self.boundaries = data["boundaries"].astype(np.float32)
        self.mean, self.var = data["obs_mean"], data["obs_var"]
        self.clip, self.eps = float(data["clip_obs"]), float(data["epsilon"])
        self.source_sha256 = str(data["source_sha256"])
        self.actors = [(
            data[f"actor_{index}_w0"], data[f"actor_{index}_b0"],
            data[f"actor_{index}_w1"], data[f"actor_{index}_b1"],
            data[f"actor_{index}_out_w"], data[f"actor_{index}_out_b"],
        ) for index in range(self.actor_count)]
        self.value = (data["value_w0"], data["value_b0"], data["value_w1"],
                      data["value_b1"], data["value_out_w"], data["value_out_b"])
        self.last_decision: dict | None = None

    def __call__(self, observation) -> int:
        raw = np.asarray(observation, dtype=np.float32)
        normalized = np.clip((raw - self.mean) / np.sqrt(self.var + self.eps),
                             -self.clip, self.clip).astype(np.float32)
        actor_index = int(np.searchsorted(self.boundaries, normalized[-1])) if self.actor_count > 1 else 0
        w0, b0, w1, b1, wo, bo = self.actors[actor_index]
        hidden_1 = np.tanh(w0 @ normalized + b0)
        hidden_2 = np.tanh(w1 @ hidden_1 + b1)
        logits = wo @ hidden_2 + bo
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        vw0, vb0, vw1, vb1, vwo, vbo = self.value
        value_hidden = np.tanh(vw0 @ normalized + vb0)
        value_hidden = np.tanh(vw1 @ value_hidden + vb1)
        estimated_value = float((vwo @ value_hidden + vbo).reshape(-1)[0])
        action = int(np.argmax(logits))
        self.last_decision = {
            "action": action,
            "actor_index": actor_index,
            "probabilities": probabilities.round(6).tolist(),
            "logits": logits.round(6).tolist(),
            "value": round(estimated_value, 6),
            "observation": raw.round(6).tolist(),
            "normalized_observation": normalized.round(6).tolist(),
            "hidden_1_summary": {
                "minimum": round(float(hidden_1.min()), 6),
                "maximum": round(float(hidden_1.max()), 6),
                "mean": round(float(hidden_1.mean()), 6),
            },
            "hidden_2_summary": {
                "minimum": round(float(hidden_2.min()), 6),
                "maximum": round(float(hidden_2.max()), 6),
                "mean": round(float(hidden_2.mean()), 6),
            },
        }
        return action

    def describe(self) -> dict:
        return {
            "algorithm": "PPO",
            "kind": "checkpoint PPO · adaptador NumPy validado",
            "adapter_path": str(self.adapter_path),
            "adapter_sha256": sha256_file(self.adapter_path),
            "model_sha256": self.source_sha256,
            "observation_shape": [132],
            "history_steps": 4,
            "features_per_step": 33,
            "action_space": {"type": "Discrete", "size": 7},
            "architecture": {
                "policy": [132, 64, 64, 7],
                "activation": "Tanh",
                "actors": self.actor_count,
                "value": [132, 64, 64, 1],
            },
            "deterministic": True,
        }


def export_ppo_numpy(model_path: Path, vecnormalize_path: Path | None, out: Path) -> Path:
    """Exporta los pesos de la politica PPO y la normalizacion a un .npz (necesita torch y SB3)."""
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.policy.state_dict().items()}
    arrays = {"w0": sd["mlp_extractor.policy_net.0.weight"], "b0": sd["mlp_extractor.policy_net.0.bias"],
              "w1": sd["mlp_extractor.policy_net.2.weight"], "b1": sd["mlp_extractor.policy_net.2.bias"],
              "wa": sd["action_net.weight"], "ba": sd["action_net.bias"]}
    norm = None
    if vecnormalize_path and Path(vecnormalize_path).exists():
        with open(vecnormalize_path, "rb") as fh:
            norm = pickle.load(fh)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays, norm_obs=bool(norm is not None and norm.norm_obs),
             obs_mean=(norm.obs_rms.mean if norm is not None else np.zeros(20)),
             obs_var=(norm.obs_rms.var if norm is not None else np.ones(20)),
             clip_obs=(norm.clip_obs if norm is not None else np.inf), epsilon=(norm.epsilon if norm is not None else 0.0),
             source_sha256=sha256_file(model_path),
             vecnormalize_sha256=sha256_file(vecnormalize_path) if norm is not None else "")
    return out


def make_policy(name: str, model: str | None = None, vecnormalize: str | None = None):
    if name == "random":
        return RandomPolicy()
    if name == "rule":
        return RulePolicy()
    if name in ("qlearning", "qtable"):
        if not model:
            raise ValueError("--model es obligatorio para qlearning (.npz)")
        return QTablePolicy(model)
    if name == "ppo":
        if not model:
            raise ValueError("--model es obligatorio para ppo (.zip)")
        return PPOPolicy(model, vecnormalize)
    raise ValueError(f"politica desconocida: {name}")


# ------------------------------------------------- modelos de un manifiesto
class ModelUnavailable(RuntimeError):
    """Los pesos del run no estan disponibles."""


class ModelIntegrityError(RuntimeError):
    """El SHA-256 de los pesos no coincide con el manifiesto."""


def model_files(models_dir: Path, manifest: dict) -> tuple[Path, Path | None]:
    """Rutas de los pesos (y de VecNormalize) de un run dentro de models/<run_id>/."""
    model_dir = Path(models_dir) / manifest["run_id"]
    if manifest["algorithm"] == "qlearning":
        return model_dir / "qlearning.npz", None
    return model_dir / "ppo_model.zip", model_dir / "vecnormalize.pkl"


def adapter_path(models_dir: Path, run_id: str) -> Path:
    """Adaptador numpy de un PPO: catalog/adapters/<run_id>.npz (versionado en Git)."""
    return Path(models_dir).parent / "catalog" / "adapters" / f"{run_id}.npz"


def verified_policy(models_dir: Path, manifest: dict):
    """Politica de un run entrenado, solo si sus pesos coinciden con el SHA-256 del manifiesto.

    Con POLICY_BACKEND=numpy (servidor sin torch), un PPO se ejecuta con su adaptador
    numpy validado, que tambien debe declarar el SHA-256 del modelo del manifiesto."""
    run_id = manifest["run_id"]
    if manifest.get("status") != "completed" or not manifest.get("model_sha256"):
        raise ModelUnavailable(f"el run {run_id} tiene estado {manifest.get('status')} y no tiene "
                               f"checkpoint disponible ({manifest.get('diagnosis') or 'sin diagnóstico'})")
    weights, normalization = model_files(models_dir, manifest)
    if not weights.exists():
        if manifest["algorithm"] == "ppo" and os.environ.get("POLICY_BACKEND") == "numpy":
            adapter = adapter_path(models_dir, run_id)
            evidence_path = Path(models_dir).parent / "docs/evidencia/adaptador-numpy.json"
            evidence = json.loads(evidence_path.read_text()).get(run_id, {}) if evidence_path.exists() else {}
            if not adapter.exists() or not evidence:
                raise ModelUnavailable(f"falta el adaptador validado de {run_id}")
            if (sha256_file(adapter) != evidence.get("adapter_sha256")
                    or evidence.get("model_sha256") != manifest["model_sha256"]):
                raise ModelIntegrityError(f"el adaptador de {run_id} no coincide con su evidencia")
            if not all(evidence.get("levels", {}).get(level, {}).get("identical_cases")
                       for level in ("n1", "n2", "n3")):
                raise ModelIntegrityError(f"validación incompleta del adaptador de {run_id}")
            policy = NumpyPPOPolicy(adapter)
            if policy.source_sha256 != manifest["model_sha256"]:
                raise ModelIntegrityError(f"el adaptador de {run_id} se exportó de otro modelo")
            return policy
        raise ModelUnavailable(f"los pesos de {run_id} no están en models/{run_id}/")
    actual = sha256_file(weights)
    if actual != manifest["model_sha256"]:
        raise ModelIntegrityError(f"SHA-256 de {weights.name} = {actual[:12]}…, el manifiesto dice "
                                  f"{manifest['model_sha256'][:12]}…: el modelo no es el registrado")
    if manifest["algorithm"] == "qlearning":
        return QTablePolicy(weights)
    if os.environ.get("POLICY_BACKEND") == "numpy":
        adapter = adapter_path(models_dir, run_id)
        if not adapter.exists():
            raise ModelUnavailable(f"falta el adaptador numpy de {run_id} (scripts/export_numpy_policies.py)")
        policy = NumpyPPOPolicy(adapter)
        if policy.source_sha256 != manifest["model_sha256"]:
            raise ModelIntegrityError(f"el adaptador de {run_id} se exportó de otro modelo")
        return policy
    return PPOPolicy(weights, normalization if normalization and normalization.exists() else None)
