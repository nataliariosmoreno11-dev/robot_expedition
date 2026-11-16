"""One process per PPO run, using the existing MLflow/DVC training pipeline."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from api.training import ROOT, read, write
from pipelines.train import configure_tracking, read_config, sha256_file, train_seed


def main(directory):
    state_path = directory / "state.json"
    state = read(state_path)
    state["pid"] = os.getpid()
    write(state_path, state)
    try:
        # Capture tracked and new source files, including uncommitted changes.
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
        sources = {}
        for name in names:
            if name.endswith((".py", ".ts", ".json", ".yaml", ".html", ".css", ".ini", ".txt", ".md", ".toml")) and not name.startswith(("catalog/", "manifests/")):
                path = ROOT / name
                if path.is_file():
                    data = path.read_bytes()
                    sources[name] = hashlib.sha256(data).hexdigest()
                    target = directory / "source" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        write(directory / "source-sha256.json", sources)
        configure_tracking()
        config_path = directory / "config.json"
        config = read_config(config_path)

        def started(run_id):
            model_dir = ROOT / "models" / run_id
            shutil.copytree(directory / "source", model_dir / "source", dirs_exist_ok=True)
            shutil.copy2(directory / "source-sha256.json", model_dir / "source-sha256.json")
            state.update(run_id=run_id, status="running")
            write(state_path, state)

        train_seed(config, config_path, sha256_file(config_path), state["seed"], on_started=started)
        state["status"] = "completed"
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        # Interrupted runs remain versionable too, even without final evaluation.
        if state.get("run_id"):
            from pipelines.train import _register_dvc_model
            model_dir = ROOT / "models" / state["run_id"]
            if (model_dir / "checkpoint_state.json").exists():
                try:
                    # The shared replay store is versioned separately. Copying its
                    # entire history into every model caused quadratic disk growth.
                    db = ROOT / "logs/planetary_episodes.sqlite"
                    if db.exists():
                        write(model_dir / "episode-store.json", {
                            "path": str(db.relative_to(ROOT)), "run_id": state["run_id"],
                            "storage": "shared; representative replays and complete metrics",
                        })
                    _register_dvc_model(model_dir, state["run_id"])
                    state["dvc_pointer"] = f"models/{state['run_id']}.dvc"
                except Exception as exc:
                    state["versioning_error"] = str(exc)
        write(state_path, state)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
