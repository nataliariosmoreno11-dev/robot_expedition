import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"run_id", "algorithm", "environment_version", "git_commit", "model_uri", "model_sha256",
            "action_spec", "origin", "parent_run_id", "evaluation_suite_id", "seed", "runtime_ref"}


def test_planetary_manifests_match_the_versioned_artifacts():
    # Recalcula cada SHA-256 desde los archivos versionados y compara con los manifiestos.
    result = subprocess.run([sys.executable, "scripts/build_planetary_manifests.py", "--check"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_planetary_manifests_follow_the_required_template():
    manifests = list((ROOT / "manifests" / "planetary").glob("*.json"))
    assert len(manifests) >= 11
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED <= manifest.keys(), (path.name, REQUIRED - manifest.keys())
        assert len(manifest["model_sha256"]) == 64
        assert len(manifest["git_commit"]) == 40
        if manifest["algorithm"] == "ppo":
            assert manifest["normalization_uri"] and manifest["obs_spec"]["shape"] == [132]
