"""Validate a frozen specialist on full mission suites; export only after all pass."""
import argparse
import json
import pickle
import shutil
import sys
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from stable_baselines3 import PPO
from pipelines.train import _LivePPOPolicy, save_json, sha256_file, sha256_json, _manifest, _git_value, _register_dvc_model
from pipelines.planetary_evaluate import evaluate_planetary
from pipelines.specialist_policy import freeze_mastered
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum


def validate(parent, candidate, report_dir):
    torch.set_num_threads(2)
    parent = parent.resolve()
    report_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads((candidate.parent / "selection.json").read_text())
    if Path(selection["parent"]).resolve() != parent:
        raise ValueError("Candidate belongs to a different parent normalizer")
    model = PPO.load(candidate, device="cpu")
    with (parent / "vecnormalize.pkl").open("rb") as stream:
        normalizer = pickle.load(stream)
    normalizer.training = False
    config = yaml.safe_load((ROOT / "configs/experiments/exp-014-specialist.yaml").read_text())
    run_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    config.update(parent_run_id=f"local:{parent}", supervised_initialization={
        "candidate_sha256": sha256_file(candidate), "dataset_sha256": sha256_file(ROOT / "configs/training/level6-demonstration.npz"),
        "selection": "best bounded scenario pilot; development seeds 740000-740002 and regression 920047; final evaluation seeds disjoint",
        "pilot_report_sha256": sha256_file(candidate.parent / "report.json"),
        "selection_details": json.loads((candidate.parent / "selection.json").read_text())})
    steps = model.num_timesteps
    shutil.copy2(parent / "curriculum_state.json", report_dir / "curriculum_state.json")
    manager = CampaignCurriculum(report_dir / "curriculum_state.json", **config["environment"]["curriculum_settings"])
    if manager.current_level != 6 or manager.mastered != [1,2,3,4,5]:
        raise ValueError("This validation is specifically for the level-six transfer")
    config["environment"].update(run_id=run_id, policy_version=run_id, record_episodes=False)
    reports = []
    policy = _LivePPOPolicy(model, normalizer)
    for level in [6,1,2,3,4,5]:
        report = evaluate_planetary(policy, config["environment"], episodes=50,
                                    seed_start=930000 if level==6 else 940000, scenario_level=level,
                                    abort_below_success_rate=manager.success_threshold)
        report.update(training_step=steps, policy_version=run_id)
        passed = manager.evaluate_scenario_mastery(report)
        save_json(report_dir / f"evaluation-{level}.json", report)
        reports.append(report)
        print(json.dumps({"level": level, "passed": passed, "metrics": report["metrics"]}), flush=True)
        if not passed:
            raise RuntimeError("Candidate rejected: safe parent and campaign unchanged")
    manager.advance_scenario()
    freeze_mastered(model, manager.mastered)
    directory = ROOT / "models" / run_id
    directory.mkdir(exist_ok=False)
    model.save(directory / "ppo_model.zip")
    shutil.copy2(parent / "vecnormalize.pkl", directory / "vecnormalize.pkl")
    torch.save(model.policy.optimizer.state_dict(), directory / "optimizer.pt")
    save_json(directory / "checkpoint_state.json", {"total_timesteps": steps})
    save_json(directory / "progress.json", {"total_timesteps": steps, "updates": model._n_updates,
              "method": "supervised initialization of isolated PPO actor; environment steps unchanged"})
    shutil.copy2(report_dir / "curriculum_state.json", directory / "curriculum_state.json")
    config["environment"]["curriculum_state"] = str(directory / "curriculum_state.json")
    save_json(directory / "config.json", config)
    for report in reports:
        name = f"periodic-evaluation-{steps}.json" if report["scenario_level"] == 6 else f"retention-{report['scenario_level']}-{steps}.json"
        save_json(directory / name, report)
    shutil.copy2(candidate.parent / "report.json", directory / "imitation-pilot.json")
    shutil.copy2(candidate.parent / "selection.json", directory / "imitation-selection.json")
    source = directory / "source"
    for filename in ("pipelines/specialist_policy.py", "scripts/pilot_specialist.py", "scripts/pilot_scenarios.py", "scripts/validate_specialist.py", "scripts/build_stage_bank.py", "configs/experiments/exp-014-specialist.yaml", "configs/training/level6-demonstration.npz", "configs/training/level2-demonstration.npz", "pipelines/train.py", "pipelines/planetary_evaluate.py", "robot_expedition/planetary/reference.py"):
        dest = source / filename; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, dest)
    save_json(directory / "sha256.json", {name: sha256_file(directory / name) for name in
              ("ppo_model.zip", "vecnormalize.pkl", "optimizer.pt", "checkpoint_state.json")})
    save_json(directory / "source-sha256.json", {
        str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()})
    model_uri = _register_dvc_model(directory, run_id)
    manifest = _manifest(run_id=run_id, config=config, config_path=directory / "config.json",
        config_hash=sha256_json(config), source_config_hash=sha256_file(ROOT / "configs/experiments/exp-014-specialist.yaml"),
        seed=41, git_commit=_git_value("rev-parse", "HEAD"), origin=_git_value("remote", "get-url", "origin"),
        budget=0, effective_budget=0, parent_run_id=f"local:{parent}", status="completed",
        diagnosis="Supervised transfer; level 6 and retention 1-5 passed. Not a ten-level release.",
        model_uri=model_uri, model_hash=sha256_file(directory / "ppo_model.zip"),
        normalization_uri=f"local:{directory / 'vecnormalize.pkl'}", metrics=reports[0]["metrics"],
        checkpoint_uri=f"local:{directory}")
    save_json(ROOT / "manifests" / f"{run_id}.json", manifest)
    job = {"job_id": job_id, "status": "completed", "algorithm": "PPO + behavior cloning", "run_id": run_id,
           "budget": 0, "seed": 41, "parent_job_id": None, "method": "validated supervised transfer",
           "parent_run_id": parent.name}
    save_json(ROOT / "logs/training_jobs" / job_id / "state.json", job)
    save_json(ROOT / "logs/training_jobs" / job_id / "config.json", config)
    save_json(report_dir / "export.json", job)
    print(json.dumps(job), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate(args.parent, args.candidate, args.output)
