"""Bounded imitation repair with independent actors, never automatic mastery."""
import argparse
from copy import deepcopy
import json
import pickle
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from pipelines.specialist_policy import specialize_by_scenario
from pipelines.train import _LivePPOPolicy
from pipelines.planetary_evaluate import evaluate_planetary
from scripts.build_stage_bank import build


def run(parent, output, levels, candidate=None):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    with (parent / "vecnormalize.pkl").open("rb") as stream:
        normalizer = pickle.load(stream)
    normalizer.training = False
    if candidate and Path(json.loads((candidate.parent / "selection.json").read_text())["parent"]).resolve() != parent.resolve():
        raise ValueError("Candidate belongs to a different parent normalizer")
    model = PPO.load(candidate or parent / "ppo_model.zip", device="cpu")
    specialize_by_scenario(model, normalizer)
    environment = json.loads((parent / "config.json").read_text())["environment"]
    environment = {**environment, "record_episodes": False}
    reports = json.loads((candidate.parent / "report.json").read_text()) if candidate else []
    selections = json.loads((candidate.parent / "selection.json").read_text())["levels"] if candidate else {}
    for level in levels:
        dataset = ROOT / f"configs/training/level{level}-demonstration.npz"
        build(output / f"stages-{level}.json", dataset, level=level)
        data = np.load(dataset)
        x = torch.tensor(normalizer.normalize_obs(data["observations"]), device=model.device)
        y = torch.tensor(data["actions"], device=model.device)
        router = model.policy.mlp_extractor.policy_net
        actor = router.actors[level-1]
        router.frozen_levels[level-1] = False
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
        torch.manual_seed(41)
        best_score, best_weights = None, None
        for iteration in range(6):
            if iteration:
                for update in range(500):
                    indices = torch.randint(len(y), (128,))
                    # Only this scenario's actor is updated; critic and others stay unchanged.
                    loss = torch.nn.functional.cross_entropy(actor(x[indices]), y[indices])
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    optimizer.step()
            report = evaluate_planetary(_LivePPOPolicy(model, normalizer), environment,
                episodes=3, seed_start=740000, scenario_level=level)
            report["supervised_updates"] = iteration*500
            reports.append(report)
            metrics = report["metrics"]
            regression = None
            if level == 2:
                # This known failing seed is development data, NOT held-out evidence.
                regression = evaluate_planetary(_LivePPOPolicy(model, normalizer), environment,
                    episodes=1, seed_start=920047, scenario_level=2)
                report["development_regression"] = regression
            evidence = [metrics] + ([regression["metrics"]] if regression else [])
            score = (min(m["success_rate"] for m in evidence),
                     min(m["samples_collected_mean"] for m in evidence),
                     -max(m["collisions_per_episode"] for m in evidence),
                     -max(m["steps_mean"] for m in evidence))
            if best_score is None or score > best_score:
                best_score, best_weights = score, deepcopy(actor.state_dict())
                selections[str(level)] = {"updates": iteration*500, "seed":41, "score":score}
            (output / "report.json").write_text(json.dumps(reports, indent=2))
            print(json.dumps({"level":level,"updates":iteration*500,"metrics":metrics}), flush=True)
        actor.load_state_dict(best_weights)
        router.frozen_levels[level-1] = level <= 5
        model.save(output / "candidate.zip")
        (output / "selection.json").write_text(json.dumps({"levels":selections,"parent":str(parent)},indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[2,6], choices=range(1,11))
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    run(args.parent, args.output, args.levels, args.candidate)
