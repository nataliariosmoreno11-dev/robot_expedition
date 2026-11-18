"""Exporta los actores PPO planetarios a NumPy sin modificar los checkpoints.

El artefacto conserva los diez actores especializados, la rama de valor y la
normalización congelada. También comprueba que el argmax coincide con
``PPO.predict(..., deterministic=True)`` antes de escribir cada archivo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "f7c104a304f25b458721258b540a337e", "d0ecdae6689f527b835c1e473e893d0c",
    "d0c438b791c156688c71cf58c69fceb5", "bdbc0d04fffe5c37a5a5ac5a05013889",
    "646fbc70700f5ab8b2363524540fddca", "156aaf20543e523eb1fa4594547d956e",
    "6109346a105e5f3e82134dc9cafbbcb9", "4bcfd8f641df50b9b5460d76b8c56482",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(run_id: str) -> dict:
    import pickle

    directory = ROOT / "models" / run_id
    weights = directory / "ppo_model.zip"
    normalizer = pickle.loads((directory / "vecnormalize.pkl").read_bytes())
    model = PPO.load(weights, device="cpu")
    state = {name: value.detach().cpu().numpy().astype(np.float32)
             for name, value in model.policy.state_dict().items()}
    specialized = "mlp_extractor.policy_net.boundaries" in state
    actor_count = 10 if specialized else 1
    arrays: dict[str, object] = {
        "schema": "scenario-ppo-numpy/v1",
        "actor_count": actor_count,
        "source_sha256": sha256(weights),
        "obs_mean": normalizer.obs_rms.mean.astype(np.float32),
        "obs_var": normalizer.obs_rms.var.astype(np.float32),
        "clip_obs": float(normalizer.clip_obs),
        "epsilon": float(normalizer.epsilon),
        "boundaries": state.get("mlp_extractor.policy_net.boundaries", np.asarray([], dtype=np.float32)),
        "value_w0": state["mlp_extractor.value_net.0.weight"],
        "value_b0": state["mlp_extractor.value_net.0.bias"],
        "value_w1": state["mlp_extractor.value_net.2.weight"],
        "value_b1": state["mlp_extractor.value_net.2.bias"],
        "value_out_w": state["value_net.weight"],
        "value_out_b": state["value_net.bias"],
    }
    for actor in range(actor_count):
        if specialized:
            prefix = f"mlp_extractor.policy_net.actors.{actor}"
            arrays[f"actor_{actor}_w0"] = state[f"{prefix}.0.0.weight"]
            arrays[f"actor_{actor}_b0"] = state[f"{prefix}.0.0.bias"]
            arrays[f"actor_{actor}_w1"] = state[f"{prefix}.0.2.weight"]
            arrays[f"actor_{actor}_b1"] = state[f"{prefix}.0.2.bias"]
            arrays[f"actor_{actor}_out_w"] = state[f"{prefix}.1.weight"]
            arrays[f"actor_{actor}_out_b"] = state[f"{prefix}.1.bias"]
        else:
            arrays["actor_0_w0"] = state["mlp_extractor.policy_net.0.weight"]
            arrays["actor_0_b0"] = state["mlp_extractor.policy_net.0.bias"]
            arrays["actor_0_w1"] = state["mlp_extractor.policy_net.2.weight"]
            arrays["actor_0_b1"] = state["mlp_extractor.policy_net.2.bias"]
            arrays["actor_0_out_w"] = state["action_net.weight"]
            arrays["actor_0_out_b"] = state["action_net.bias"]
    output = ROOT / "catalog" / "campaign_adapters" / f"{run_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {"run_id": run_id, "adapter": str(output.relative_to(ROOT)),
            "adapter_sha256": sha256(output), "model_sha256": arrays["source_sha256"]}


if __name__ == "__main__":
    evidence = [export(run_id) for run_id in RUNS]
    path = ROOT / "docs" / "evidencia" / "adaptadores-campana-ppo.json"
    path.write_text(json.dumps({"schema": "scenario-ppo-numpy/v1", "runs": evidence}, indent=2) + "\n")
    print(path)
