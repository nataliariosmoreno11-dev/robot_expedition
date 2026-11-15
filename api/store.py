"""Catalogo de runs y carga verificada de modelos para la API.

Un run puede ser:
- un modelo entrenado, descrito por manifests/<run_id>.json (paso 3); sus pesos
  estan en models/<run_id>/ o se recuperan con `dvc pull` (API_DVC_PULL=1);
- un baseline sin pesos: baseline-random o baseline-rule (paso 2).

Antes de usar un modelo se comprueba su SHA-256 contra el manifiesto.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from pipelines.policies import (
    ModelIntegrityError, ModelUnavailable, RandomPolicy, RulePolicy, model_files, verified_policy, adapter_path,
)
from robot_expedition.envs.beacon_expedition import ENVIRONMENT_VERSION

LEVELS = ("n1", "n2", "n3")
BASELINES = {
    "baseline-random": {"algorithm": "random", "model_name": "Aleatoria (referencia)",
                        "hypothesis": "Referencia aleatoria: cota inferior del éxito"},
    "baseline-rule": {"algorithm": "rule", "model_name": "Regla (referencia)",
                      "hypothesis": "Referencia de regla: girar hacia el objetivo y esquivar"},
}


class RunNotFound(LookupError):
    pass


__all__ = ["LEVELS", "BASELINES", "ModelIntegrityError", "ModelUnavailable", "RunNotFound", "Store"]


@dataclass
class Store:
    root: Path
    _policies: dict = field(default_factory=dict)

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog" / "catalog.json"

    # ------------------------------------------------------------ catalogo
    def manifests(self) -> dict[str, dict]:
        """Manifiestos de modelos candidatos (manifests/smoke no se publica)."""
        out = {}
        for path in sorted(self.manifests_dir.glob("*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            out[manifest["run_id"]] = manifest
        return out

    def catalog(self) -> dict:
        """catalog/catalog.json si el pipeline lo publico; si no, se arma desde los manifiestos."""
        baselines = [{"run_id": run_id, "level_id": "todos", "status": "baseline",
                      "environment_version": ENVIRONMENT_VERSION, **info}
                     for run_id, info in BASELINES.items()]
        if self.catalog_path.exists():
            catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            known = {run["run_id"] for run in catalog.get("runs", [])}
            catalog["runs"] = [*catalog.get("runs", []), *(b for b in baselines if b["run_id"] not in known)]
            catalog.setdefault("source", "catalog/catalog.json")
            return catalog
        runs = []
        for manifest in self.manifests().values():
            runs.append({key: manifest.get(key) for key in (
                "run_id", "experiment_id", "level_id", "algorithm", "status", "environment_version",
                "git_commit", "seed", "budget_timesteps", "effective_budget_timesteps",
                "parent_run_id", "model_sha256", "evaluation_suite_id", "metrics", "diagnosis")}
                | {"model_name": f"{manifest.get('algorithm')} · {manifest.get('experiment_id')}"})
        return {"runs": [*runs, *baselines], "recommended_run_id": {}, "served_version": None,
                "source": "manifests (sin catálogo publicado)"}

    def run(self, run_id: str) -> dict:
        if run_id in BASELINES:
            return {"run_id": run_id, "status": "baseline", "environment_version": ENVIRONMENT_VERSION,
                    "level_id": "todos", **BASELINES[run_id]}
        manifest = self.manifests().get(run_id)
        if manifest is None:
            raise RunNotFound(run_id)
        return manifest

    # ------------------------------------------------------------- modelos
    def _dvc_pull(self, run_id: str) -> str | None:
        dvc_file = self.models_dir / f"{run_id}.dvc"
        if os.environ.get("API_DVC_PULL") != "1" or not dvc_file.exists() or not shutil.which("dvc"):
            return None
        result = subprocess.run(["dvc", "pull", str(dvc_file.relative_to(self.root))], cwd=self.root,
                                capture_output=True, text=True, timeout=300)
        return None if result.returncode == 0 else (result.stderr or result.stdout).strip()[-300:]

    def _release_pull(self, run_id: str) -> str | None:
        """Extrae models/<run_id>/ del ZIP publico de la release (API_MODELS_URL, una o varias URL)."""
        urls = [u.strip() for u in os.environ.get("API_MODELS_URL", "").split(",") if u.strip()]
        error = None
        for url in urls:
            cache = self.models_dir / f".release-{hashlib.sha256(url.encode()).hexdigest()[:12]}.zip"
            try:
                if not cache.exists():
                    self.models_dir.mkdir(parents=True, exist_ok=True)
                    with urllib.request.urlopen(url, timeout=120) as response, open(cache.with_suffix(".tmp"), "wb") as fh:
                        shutil.copyfileobj(response, fh)
                    cache.with_suffix(".tmp").replace(cache)
                prefix = f"models/{run_id}/"
                with zipfile.ZipFile(cache) as archive:
                    for member in archive.infolist():
                        name = member.filename.replace("\\", "/")      # ZIP armados en Windows
                        if not name.startswith(prefix) or member.is_dir() or ".." in name.split("/"):
                            continue
                        target = self.root / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                if (self.models_dir / run_id).exists():
                    return None
            except (OSError, zipfile.BadZipFile, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return error

    def policy(self, run_id: str):
        """Politica lista para actuar; los modelos se verifican y se guardan en memoria."""
        if run_id == "baseline-random":
            return RandomPolicy()
        if run_id == "baseline-rule":
            return RulePolicy()
        manifest = self.run(run_id)
        cached = self._policies.get(run_id)
        if cached is not None:
            return cached
        weights, _ = model_files(self.models_dir, manifest)
        if (manifest.get("algorithm") == "ppo" and os.environ.get("POLICY_BACKEND") == "numpy"
                and adapter_path(self.models_dir, run_id).exists()):
            policy = verified_policy(self.models_dir, manifest)
            self._policies[run_id] = policy
            return policy
        if manifest.get("status") == "completed" and not weights.exists():
            error = self._dvc_pull(run_id)
            if not weights.exists():
                error = self._release_pull(run_id) or error
            if not weights.exists():
                raise ModelUnavailable(f"los pesos de {run_id} no están en models/{run_id}/ y no se pudieron "
                                       f"recuperar con DVC ni de la release" + (f": {error}" if error else
                                       " (define API_MODELS_URL o API_DVC_PULL=1 con las credenciales del remoto)"))
        policy = verified_policy(self.models_dir, manifest)
        self._policies[run_id] = policy
        return policy
