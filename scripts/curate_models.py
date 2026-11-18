"""Export one verified milestone per scenario count; optionally prune obsolete weights.

Run with python -m scripts.curate_models [--prune]. Reports and job history remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"ppo_model.zip", "optimizer.pt", "vecnormalize.pkl", "qlearning.npz", "sarsa.npz"}
REQUIRED = ("ppo_model.zip", "optimizer.pt", "vecnormalize.pkl", "checkpoint_state.json")


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verified_levels(directory):
    if not all((directory / name).is_file() for name in REQUIRED):
        return []
    state = read(directory / "curriculum_state.json")
    step = read(directory / "checkpoint_state.json").get("total_timesteps")
    levels = [n for n in state.get("mastered", []) if 1 <= n <= 8]
    evaluations = state.get("evaluations", {})
    if not levels or step is None:
        return []
    for level in levels:
        result = evaluations.get(str(level), {})
        # Inherited approval from older weights cannot certify this checkpoint.
        if (not result.get("passed") or result.get("episodes", 0) < 50
                or result.get("success_rate", 0) < .85 or result.get("training_step") != step):
            return []
    return sorted(levels)


def curate(prune=False):
    models = (ROOT / "models").resolve()
    jobs = ROOT / "logs/training_jobs"
    candidates = list(models.glob("*/mastered-scenario-*"))
    # Independently audited six-scenario export has checksums and 300 evaluations.
    audited = read(ROOT / "docs/level6-recovery-result.json")
    if audited:
        source = models / audited["run_id"]
        weights = source / "ppo_model.zip"
        if weights.exists() and hashlib.sha256(weights.read_bytes()).hexdigest() == audited["weights_sha256"]:
            candidates.append(source)
    candidates += [p.parent for p in models.glob("*/approved-version.json")]
    chosen = {}
    for source in candidates:
        levels = verified_levels(source)
        if not levels:
            continue
        step = read(source / "checkpoint_state.json")["total_timesteps"]
        key = tuple(levels)
        if key not in chosen or step > chosen[key][0]:
            chosen[key] = (step, source)
    catalog = []
    for levels, (step, source) in sorted(chosen.items()):
        original = read(source / "approved-version.json").get("source", str(source.relative_to(ROOT)))
        run_id = uuid.uuid5(uuid.NAMESPACE_URL, "robot-expedition:" + original).hex
        target = models / run_id
        if target != source:
            target.mkdir(exist_ok=True)
            for name in (*REQUIRED, "curriculum_state.json"):
                shutil.copy2(source / name, target / name)
            parent = source.parent if source.name.startswith("mastered-") else source
            shutil.copy2(parent / "config.json", target / "config.json")
        # Preserve full original evaluation cases alongside the exact saved weights.
        evidence_source = ROOT / original
        evidence_parent = evidence_source.parent if evidence_source.name.startswith("mastered-") else evidence_source
        evidence = target / "evidence"
        evidence.mkdir(exist_ok=True)
        for report_path in evidence_parent.glob("*.json"):
            report = read(report_path)
            if (isinstance(report, dict) and report.get("training_step") == step
                    and report.get("scenario_level") in levels and len(report.get("cases", [])) >= 50):
                shutil.copy2(report_path, evidence / report_path.name)
        hashes = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in REQUIRED}
        write(target / "sha256.json", hashes)
        entry = {"run_id": run_id, "job_id": run_id, "completed_scenarios": list(levels),
                 "source": original, "total_timesteps": step, "weights_sha256": hashes["ppo_model.zip"]}
        write(target / "approved-version.json", entry)
        write(target / "progress.json", {"total_timesteps": step})
        write(jobs / run_id / "state.json", {"job_id": run_id, "run_id": run_id,
              "status": "completed", "algorithm": "PPO", "budget": 0, "parent_job_id": None})
        catalog.append(entry)
    protected = {entry["run_id"] for entry in catalog}
    campaign = read(ROOT / "logs/autonomous_campaign/state.json")
    for key in ("active_job_id", "parent_job_id"):
        if campaign.get(key):
            state = read(jobs / campaign[key] / "state.json")
            protected.add(state.get("run_id"))
            if state.get("parent_job_id"):
                protected.add(read(jobs / state["parent_job_id"] / "state.json").get("run_id"))
    # Keep the audited original and any live worker plus its immediate resume parent.
    if audited:
        protected.add(audited["run_id"])
    for path in jobs.glob("*/state.json"):
        state = read(path)
        if state.get("status") in ("starting", "running", "stopping"):
            protected.add(state.get("run_id"))
            parent = state.get("parent_job_id")
            if parent:
                protected.add(read(jobs / parent / "state.json").get("run_id"))
    release = read(models / "final-release.json")
    protected.add(release.get("run_id"))
    removed = []
    for directory in models.iterdir():
        if not directory.is_dir() or directory.name in protected:
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.name not in WEIGHTS:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(models) or path.is_symlink():
                raise ValueError(f"Unsafe cleanup path: {path}")
            removed.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
            if prune:
                path.unlink()
    report = {"versions": catalog, "protected_runs": sorted(x for x in protected if x),
              "pruned": prune, "bytes": sum(x["bytes"] for x in removed), "files": removed}
    write(ROOT / "logs/model-cleanup.json", report)
    write(ROOT / "docs/approved-models.json", {"versions": catalog})
    print(json.dumps({"versions": catalog, "pruned": prune, "files": len(removed),
                      "freed_mb": round(report["bytes"] / 1024**2, 2)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", action="store_true")
    curate(parser.parse_args().prune)
