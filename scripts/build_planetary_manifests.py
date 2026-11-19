"""Genera los manifiestos de los modelos planetarios (PPO de la campaña y Q-learning).

Cada manifiesto sigue la plantilla de la sección 11 del enunciado: pesos y SHA-256,
contrato de observaciones/acciones, ambiente compatible, configuración y su hash,
run, presupuesto, semillas, origen, optimizador, evaluación y URI inmutable.

Los hashes se calculan de los archivos reales y se contrastan con los que registró
el entrenamiento (sha256.json / approved-version.json): si no coinciden, el script
falla en vez de escribir un manifiesto incorrecto. Las URI apuntan al archivo dentro
del commit exacto en que entró al repositorio, así que no cambian. El
``environment_hash`` se calcula con el código del ambiente de ese mismo commit
(no con el de HEAD): un cambio posterior del ambiente no reescribe la procedencia
del modelo, y la API puede detectar que el ambiente actual ya no es el de entrenamiento.

Uso:
    python scripts/build_planetary_manifests.py            # escribe manifests/planetary/*.json
    python scripts/build_planetary_manifests.py --check    # solo verifica (para CI/pipeline)
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = "https://github.com/nataliariosmoreno11-dev/robot_expedition"
OUT = ROOT / "manifests" / "planetary"

# Hitos aprobados de la campaña PPO (uno por escenario superado) y el modelo final.
CAMPAIGN_RUNS = (
    "f7c104a304f25b458721258b540a337e",
    "d0ecdae6689f527b835c1e473e893d0c",
    "d0c438b791c156688c71cf58c69fceb5",
    "bdbc0d04fffe5c37a5a5ac5a05013889",
    "646fbc70700f5ab8b2363524540fddca",
    "156aaf20543e523eb1fa4594547d956e",
    "6109346a105e5f3e82134dc9cafbbcb9",
    "4bcfd8f641df50b9b5460d76b8c56482",
    "f59726370d0d471a9790482135f4e3e2",
)

OBSERVATION_COMPONENTS = [
    *[{"name": f"lidar_{i}", "unit": "m", "range": [0.0, 5.0],
       "note": "rayo horizontal a 0.3 m; ángulos de -90° a +90° (izquierda a derecha)"} for i in range(9)],
    {"name": "navigation_distance", "unit": "m", "range": [0.0, 17.0],
     "note": "a la muestra objetivo visible o a la base si ya se recolectó todo; 5.0 si no hay"},
    {"name": "relative_sin", "unit": "1", "range": [-1.0, 1.0], "note": "0 si no hay objetivo visible"},
    {"name": "relative_cos", "unit": "1", "range": [-1.0, 1.0], "note": "0 si no hay objetivo visible"},
    {"name": "forward_speed", "unit": "fracción de 0.6 m/s", "range": [-1.0, 1.0]},
    {"name": "turn_rate", "unit": "fracción de 1.2 rad/s", "range": [-1.0, 1.0]},
    {"name": "battery", "unit": "fracción de 100 %", "range": [0.0, 1.0]},
    {"name": "collected_fraction", "unit": "1", "range": [0.0, 1.0]},
    {"name": "remaining_fraction", "unit": "1", "range": [0.0, 1.0]},
    {"name": "inventory_fraction", "unit": "1", "range": [0.0, 1.0]},
    *[{"name": f"target_{kind}", "unit": "one-hot", "range": [0.0, 1.0]}
      for kind in ("basalto", "hematita", "sulfato")],
    {"name": "sample_visible", "unit": "booleano", "range": [0.0, 1.0]},
    {"name": "visible_distance", "unit": "m", "range": [0.0, 17.0]},
    *[{"name": f"visible_{kind}", "unit": "one-hot", "range": [0.0, 1.0]}
      for kind in ("basalto", "hematita", "sulfato")],
    {"name": "gravity", "unit": "fracción de 9.81 m/s²", "range": [0.0, 1.5]},
    {"name": "local_traction", "unit": "coeficiente", "range": [0.0, 1.5]},
    {"name": "visibility", "unit": "fracción de 5 m", "range": [0.0, 1.0]},
    {"name": "wind", "unit": "fracción de 30 m/s", "range": [0.0, 1.0]},
    {"name": "pitch", "unit": "rad", "range": [-3.1416, 3.1416]},
    {"name": "roll", "unit": "rad", "range": [-3.1416, 3.1416]},
    {"name": "difficulty_level", "unit": "fracción de 8 (10 en campaña)", "range": [0.0, 1.0]},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def introduced_in(path: Path) -> str:
    """SHA completo del primer commit que contiene el archivo (identidad inmutable)."""
    relative = path.relative_to(ROOT).as_posix()
    shas = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", relative],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    if not shas:
        raise SystemExit(f"{relative} no está versionado en Git: no hay URI inmutable")
    return shas[-1]


def full_sha(ref: str) -> str:
    return subprocess.run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


_SNAPSHOTS: dict[str, Path] = {}
_HASH_AT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from robot_expedition.envs.planetary_exploration import planetary_environment_hash
print(planetary_environment_hash(json.loads(sys.stdin.read())))
"""


def environment_hash_at(commit: str, environment: dict | None) -> str:
    """Hash del ambiente planetario según el código versionado en ``commit``."""
    if commit not in _SNAPSHOTS:
        folder = Path(tempfile.mkdtemp(prefix=f"env-{commit[:8]}-"))
        archive = subprocess.run(["git", "archive", "--format=tar", commit],
                                 cwd=ROOT, capture_output=True, check=True).stdout
        tar_path = folder / "src.tar"
        tar_path.write_bytes(archive)
        with tarfile.open(tar_path) as tar:
            tar.extractall(folder, filter="data")
        _SNAPSHOTS[commit] = folder
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
    result = subprocess.run([sys.executable, "-c", _HASH_AT, str(_SNAPSHOTS[commit])],
                            cwd=_SNAPSHOTS[commit], input=json.dumps(environment or {}),
                            capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()[-1]


def artifact(path: Path) -> dict:
    commit = introduced_in(path)
    return {"uri": f"{REPO}/raw/{commit}/{path.relative_to(ROOT).as_posix()}",
            "sha256": sha256(path), "git_commit": commit}


def observation_spec(history_steps: int = 4) -> dict:
    return {
        "shape": [OBSERVATION_COMPONENTS.__len__() * history_steps],
        "layout": f"{history_steps} lecturas apiladas de {len(OBSERVATION_COMPONENTS)} componentes; "
                  "la más reciente al final",
        "components": OBSERVATION_COMPONENTS,
        "scaling": "VecNormalize (media/varianza congeladas) antes de la red; clip ±10",
    }


def action_spec() -> dict:
    from robot_expedition.envs.planetary_exploration import PLANETARY_ACTIONS
    return {"type": "Discrete", "n": len(PLANETARY_ACTIONS),
            "actions": [{"id": i, "name": name, "linear_m_s": v, "angular_rad_s": w}
                        for i, (name, v, w) in enumerate(PLANETARY_ACTIONS)],
            "decision_dt_s": 0.5}


def campaign_manifest(run_id: str) -> dict:
    from robot_expedition.envs.planetary_exploration import (
        PLANETARY_ENVIRONMENT_VERSION)
    folder = ROOT / "models" / run_id
    approved = json.loads((folder / "approved-version.json").read_text())
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    recorded = json.loads((folder / "sha256.json").read_text()) if (folder / "sha256.json").exists() else {}

    weights = artifact(folder / "ppo_model.zip")
    expected = recorded.get("ppo_model.zip") or approved.get("weights_sha256")
    if expected and expected != weights["sha256"]:
        raise SystemExit(f"{run_id}: el SHA-256 de ppo_model.zip no coincide con el registrado")
    normalization = artifact(folder / "vecnormalize.pkl")
    if recorded.get("vecnormalize.pkl") not in (None, normalization["sha256"]):
        raise SystemExit(f"{run_id}: el SHA-256 de vecnormalize.pkl no coincide")
    adapter_path = ROOT / "catalog" / "campaign_adapters" / f"{run_id}.npz"
    optimizer_path = folder / "optimizer.pt"

    parent = config.get("parent_run_id")
    parent_run = Path(str(parent)).name if parent else None
    completed = approved.get("completed_scenarios", [])
    contract = approved.get("environment_contract", "campaign/dynamic-storms")
    return {
        "run_id": run_id,
        "algorithm": "ppo",
        "model_family": "planetary-campaign-ppo",
        "level_id": f"campaña L1-L{max(completed)}" if completed else None,
        "status": "approved",
        "completed_scenarios": completed,
        "experiment_id": config.get("experiment_id"),
        "hypothesis": config.get("hypothesis"),
        "change": config.get("change"),
        "environment_version": PLANETARY_ENVIRONMENT_VERSION,
        "environment_contract": contract,
        "environment_hash": environment_hash_at(weights["git_commit"], config.get("environment"))
                            if config else None,
        "environment_hash_source": f"código del ambiente en {weights['git_commit'][:7]} (commit que versionó el modelo)"
                                   if config else None,
        "git_commit": weights["git_commit"],
        "config_uri": artifact(config_path)["uri"] if config else None,
        "config_sha256": sha256(config_path) if config else None,
        "model_uri": weights["uri"],
        "model_sha256": weights["sha256"],
        "normalization_uri": normalization["uri"],
        "normalization_sha256": normalization["sha256"],
        "serving_adapter_uri": artifact(adapter_path)["uri"] if adapter_path.exists() else None,
        "serving_adapter_sha256": sha256(adapter_path) if adapter_path.exists() else None,
        "optimizer_uri": artifact(optimizer_path)["uri"] if optimizer_path.exists() else None,
        "optimizer_sha256": sha256(optimizer_path) if optimizer_path.exists() else None,
        "runtime_ref": "Python 3.12.8 · requirements.txt (SB3/PyTorch para continuar) · "
                       "api/requirements-lite.txt con POLICY_BACKEND=numpy para servir",
        "hyperparams": config.get("hyperparams"),
        "seed": config.get("seed"),
        "seeds": config.get("seeds"),
        "budget_timesteps": config.get("budget_timesteps"),
        "effective_timesteps": approved.get("total_timesteps"),
        "origin": ("continuación de checkpoint" if parent_run else "desde cero") if config
                  else "no registrado: el config.json de este run no está en el repositorio",
        "parent_run_id": parent_run,
        "evaluation_suite_id": approved.get("validation")
                               or f"campaign scenario_01..08 · seed {config.get('periodic_evaluation_seed_start', 910000)}",
        "evaluation_evidence": sorted(p.relative_to(ROOT).as_posix() for p in (folder / "evidence").glob("*.json")),
        "obs_spec": observation_spec(int((config.get("environment") or {}).get("history_steps", 4))),
        "action_spec": action_spec(),
    }


def qlearning_manifests() -> list[dict]:
    from robot_expedition.envs.planetary_exploration import PLANETARY_ENVIRONMENT_VERSION
    common = {
        "algorithm": "qlearning",
        "model_family": "planetary-qlearning-tabular",
        "environment_version": PLANETARY_ENVIRONMENT_VERSION,
        "runtime_ref": "Python 3.12.8 · api/requirements-lite.txt (NumPy)",
        "action_spec": action_spec(),
        "hyperparams": {"alpha": 0.14, "gamma": 0.98, "epsilon_start": 0.75, "epsilon_min": 0.06,
                        "epsilon_decay": 0.992, "guide_probability": 0.95, "warmup_visits": 6},
    }
    release = f"{REPO}/releases/download"
    return [
        {**common,
         "run_id": "qlearning-seed-v2",
         "status": "recommended",
         "level_id": "planetario L1-L8",
         "hypothesis": "Si la tabla Q aprende sobre la sugerencia de un guía planificador (estado + acción "
                       "sugerida), resuelve L2-L8, que el guía reactivo no resolvía.",
         "change": "Estado v2 con la acción del guía; límites de pasos del escenario; sin piso de recompensa; "
                   "observación más reciente; escudo de seguridad junto a peligros.",
         "git_commit": full_sha("ce0779e"),
         "model_uri": f"{release}/qlearning-seed-v2/planetary_qlearning.npz",
         "model_sha256": "1e203e0d39cde8c5f4ca2291af5e9cbcf8b5b0bb0d92ad96e2c0d4d64f99ac5c",
         "normalization_uri": None,
         "state_spec": {"size": 12, "discretization": "robot_expedition/planetary/qlearning.py:discretize_planetary",
                        "note": "11 componentes discretizados de la observación más reciente + acción sugerida "
                                "por el guía; ver docs/auditoria-qlearning-planetario.md#discretización"},
         "effective_episodes": 480, "effective_updates": 124299, "states": 8294,
         "seed": 1001, "seeds": "1001-1480 (una por episodio de entrenamiento)",
         "origin": "desde cero",
         "parent_run_id": None,
         "training_command": "python scripts/train_planetary_qlearning.py --rounds 12 --seeds-per-level 5",
         "evaluation_suite_id": "planetario L1-L8 · semillas 6000-6009 · límites de diseño",
         "evaluation_evidence": ["docs/evidencia/evaluacion-planificador-qlearning.json"],
         "evaluation_summary": {"success": 77, "episodes": 80, "code_commit": full_sha("e355343")}},
        {**common,
         "run_id": "qlearning-seed-v1",
         "status": "historical · incompatible con el estado v2",
         "level_id": "planetario L1-L8",
         "hypothesis": "Un checkpoint base entrenado offline evita que producción arranque con la tabla vacía.",
         "change": "Primer checkpoint base (estado v1 de 11 componentes, guía reactivo, 50/100 pasos).",
         "git_commit": full_sha("f558fbb"),
         "model_uri": f"{release}/qlearning-seed-v1/planetary_qlearning.npz",
         "model_sha256": "7adb5883f3734a20829f515f591b2484e214c7831e0c298f83e8be7814d26af9",
         "normalization_uri": None,
         "state_spec": {"size": 11, "note": "estado v1 sin acción del guía; la API lo conserva renombrado"},
         "effective_episodes": 2400, "effective_updates": 112970, "states": 1309,
         "seed": 1001, "seeds": "1001-3400",
         "origin": "desde cero", "parent_run_id": None,
         "training_command": "python scripts/train_planetary_qlearning.py --rounds 30 --seeds-per-level 10",
         "evaluation_suite_id": "planetario L1-L8 · semillas 5000+",
         "evaluation_summary": {"note": "L1 ~70 %; L2-L8 0 % (docs/auditoria-qlearning-planetario.md)"}},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verifica sin escribir")
    args = parser.parse_args()
    manifests = [campaign_manifest(run_id) for run_id in CAMPAIGN_RUNS] + qlearning_manifests()
    OUT.mkdir(parents=True, exist_ok=True)
    changed = []
    for manifest in manifests:
        path = OUT / f"{manifest['run_id']}.json"
        text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            changed.append(path.name)
            if not args.check:
                path.write_text(text, encoding="utf-8")
    print(f"{len(manifests)} manifiestos; {'desactualizados' if args.check else 'escritos'}: {changed or 'ninguno'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
