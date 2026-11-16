"""Bounded specialist/behavior-cloning pilot; never promotes curriculum."""
import argparse
import sys, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from pipelines.specialist_policy import specialize
from pipelines.train import _LivePPOPolicy
from pipelines.planetary_evaluate import evaluate_planetary
from scripts.build_stage_bank import build


def run(parent, output):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / "configs/training/level6-demonstration.npz"
    build(output / "stages.json", dataset)
    with (parent / "vecnormalize.pkl").open("rb") as f:
        normalizer = pickle.load(f)
    normalizer.training = False
    model = PPO.load(parent / "ppo_model.zip", device="cpu")
    specialize(model, normalizer)
    config = json.loads((parent / "config.json").read_text())
    environment = {**config["environment"], "record_episodes": False}
    data = np.load(dataset)
    x = torch.tensor(normalizer.normalize_obs(data["observations"]), device=model.device)
    y = torch.tensor(data["actions"], device=model.device)
    optimizer = torch.optim.Adam(model.policy.mlp_extractor.policy_net.specialist.parameters(), lr=0.001)
    reports = []
    best = (-1, -1, float('-inf'))
    for iteration in range(6):
        if iteration:
            for update in range(500):
                indices = torch.randint(len(y), (128,))
                loss = -model.policy.get_distribution(x[indices]).log_prob(y[indices]).mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
        report = evaluate_planetary(_LivePPOPolicy(model, normalizer), environment,
                                    episodes=3, seed_start=730000, scenario_level=6)
        report["supervised_updates"] = iteration * 500
        reports.append(report)
        (output / "report.json").write_text(json.dumps(reports, indent=2))
        m = report["metrics"]
        score = (m["success_rate"], m["samples_collected_mean"], -m["collisions_per_episode"])
        if score > best:
            best = score
            model.save(output / "candidate.zip")
            (output / "selection.json").write_text(json.dumps({"updates": iteration*500, "seed":41, "parent":str(parent), "score":score}))
        print(json.dumps({"updates":iteration*500, "metrics":m}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.parent, args.output)
