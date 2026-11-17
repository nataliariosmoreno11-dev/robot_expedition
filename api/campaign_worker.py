"""Detached, resumable campaign supervisor. No rendering or browser required.

Run: python -m api.campaign_worker --parent-job <id>
Stop: POST /training/campaign/stop (or create logs/autonomous_campaign/stop).
"""
import argparse
import fcntl
import os
import shutil
import sqlite3
import time
from pathlib import Path

from api import training as jobs


def qualifies(report):
    """Require actual distinct cases for every level, not aggregate success."""
    cases = report.get("cases", [])
    for level in range(1, 11):
        rows = [row for row in cases if row.get("difficulty_level") == level]
        if len({row["seed"] for row in rows}) < 50:
            return False
        if sum(bool(row["success"]) for row in rows) / len(rows) < .85:
            return False
        if sum(row["collisions"] for row in rows) / len(rows) > 2:
            return False
    return True


def certify(job_id):
    from pipelines.policies import PPOPolicy
    from pipelines.planetary_evaluate import evaluate_planetary
    from pipelines.train import _register_dvc_model

    model_dir = jobs.checkpoint_for(job_id)
    policy = PPOPolicy(model_dir / "ppo_model.zip", model_dir / "vecnormalize.pkl")
    config = jobs.read(model_dir / "config.json")
    report = evaluate_planetary(policy, {
        **config["environment"], "curriculum": False,
        "run_id": model_dir.name, "policy_version": f"final-candidate:{model_dir.name}",
        "episode_db": str(jobs.ROOT / "logs/planetary_episodes.sqlite"), "record_episodes": True,
    }, episodes=500, seed_start=2_100_000, max_level=10)
    report.update(purpose="final publication gate: one frozen policy, all ten scenarios",
                  job_id=job_id, hashes=jobs.read(model_dir / "sha256.json"))
    jobs.write(model_dir / "public-evaluation.json", report)
    # Include the final evaluation's recorded steps in the versioned snapshot.
    with sqlite3.connect(jobs.ROOT / "logs/planetary_episodes.sqlite") as source:
        with sqlite3.connect(model_dir / "episodes.sqlite") as target:
            source.backup(target)
    _register_dvc_model(model_dir, model_dir.name)
    if not qualifies(report):
        return None
    release = {"job_id": job_id, "run_id": model_dir.name, "published_at": time.time(),
               "scenario_count": 10, "episodes_per_scenario": 50,
               "evaluation": report["metrics"], "hashes": report["hashes"],
               "report": str((model_dir / "public-evaluation.json").relative_to(jobs.ROOT)),
               "dvc_pointer": f"models/{model_dir.name}.dvc"}
    jobs.write(jobs.ROOT / "models/final-release.json", release)
    return release


def run(parent_job, budget, resume=False):
    directory = jobs.ROOT / "logs/autonomous_campaign"
    directory.mkdir(parents=True, exist_ok=True)
    # A second supervisor cannot race the current one, even across API restarts.
    with (directory / "supervisor.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = jobs.read(directory / "state.json", {})
        if state.get("status") == "completed":
            return
        if (directory / "stop").exists() and not resume:
            raise RuntimeError("Campaign stopped explicitly; remove its stop file to resume")
        if resume:
            active = state.get("active_job_id")
            if active:
                previous = jobs.read(jobs.job_path(active) / "state.json")
                if previous["status"] == "interrupted":
                    jobs.checkpoint_for(active)
                    state.setdefault("rounds", []).append({"job_id": active, "status": "interrupted"})
                    state.update(parent_job_id=active, active_job_id=None)
            (directory / "stop").unlink(missing_ok=True)
        os.environ.update(ENABLE_LOCAL_TRAINING="1", EXPEDITION_CAMPAIGN_WORKER="1",
                          EXPEDITION_STOP_FILE=str(directory / "stop"))
        state.update(pid=os.getpid(), status="running", budget_per_run=budget)
        state.pop("error", None)
        state.setdefault("parent_job_id", parent_job)
        state.setdefault("rounds", [])
        def save():
            state["updated_at"] = time.time()
            jobs.write(directory / "state.json", state)
        save()
        try:
            while not (directory / "stop").exists():
                disks = [jobs.ROOT]
                if Path('/mnt/c').is_dir():
                    disks.append(Path('/mnt/c'))
                if min(shutil.disk_usage(path).free for path in disks) < 10 * 1024**3:
                    if state.get("active_job_id"):
                        jobs.stop(state["active_job_id"])
                    raise RuntimeError("Less than 10 GiB free on Linux or Windows C:; stopping to save checkpoints")
                job_id = state.get("active_job_id")
                if not job_id:
                    job = jobs.start(jobs.StartTraining(budget=budget, seed=1 + len(state["rounds"]),
                                                       parent_job_id=state["parent_job_id"]))
                    job_id = job["job_id"]
                    state.update(active_job_id=job_id, status="running")
                    save()
                job = jobs.describe(jobs.job_path(job_id))
                model_dir = jobs.ROOT / "models" / (job.get("run_id") or "pending")
                curriculum = jobs.read(model_dir / "curriculum_state.json", {})
                state.update(acceleration=job.get("acceleration"), progress=job["progress"], mastered=curriculum.get("mastered", []),
                             current_level=curriculum.get("current_level", 1))
                save()
                if job["status"] in ("starting", "running", "stopping"):
                    time.sleep(5)
                    continue
                if job["status"] != "completed":
                    raise RuntimeError(f"Job {job_id}: {job['status']}; {job.get('error', 'check worker.log')}")
                # The worker writes completed before its interpreter exits.
                # Wait for that exit before trying to acquire the next job slot.
                process = jobs.processes.get(job_id)
                if process is not None and process.poll() is None:
                    time.sleep(1)
                    continue
                if set(curriculum.get("mastered", [])) == set(range(1, 11)):
                    state["status"] = "validating"
                    save()
                    release = certify(job_id)
                    if release:
                        state.update(status="completed", release=release)
                        save()
                        return
                state["rounds"].append({"job_id": job_id, "run_id": job["run_id"],
                                        "progress": job["progress"], "evaluation": job["evaluation"]})
                state.update(parent_job_id=job_id, active_job_id=None, status="running")
                save()
            state["status"] = "stopped"
        except KeyboardInterrupt:
            state["status"] = "stopped"
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if state.get("status") == "stopped" and state.get("active_job_id"):
                jobs.stop(state["active_job_id"])
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--budget", type=int, default=262144)
    parser.add_argument("--resume", action="store_true", help="Resume after an explicit stop, keeping saved weights")
    args = parser.parse_args()
    run(args.parent_job, args.budget, resume=args.resume)
