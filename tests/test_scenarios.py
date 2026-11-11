from pathlib import Path

import yaml

from robot_expedition import BeaconExpeditionEnv
from robot_expedition.envs.beacon_expedition import environment_hash


def test_varied_experiment_uses_generalization_suite():
    config = yaml.safe_load(Path("configs/experiments/exp-006.yaml").read_text(encoding="utf-8"))
    env = BeaconExpeditionEnv(level=config["level_id"], **config["environment"])
    _, info = env.reset(seed=0)
    assert info["scenario_id"] == "easy-open-north"
    assert config["evaluation_suite_id"] == "suite-n2-generalization-v1"
    assert environment_hash("n2") != environment_hash("n2", config["environment"])
    env.close()
