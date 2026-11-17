import numpy as np
from pipelines.accelerated_curriculum import stage_indices
from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv


def test_short_final_stage_is_not_drowned_by_long_search_stage():
    stages = np.array([0]*1000+[1]*20+[2]*2)
    indices = stage_indices(stages,np.random.default_rng(4),size=3000)
    counts = np.bincount(stages[indices])
    assert all(800 < count < 1200 for count in counts)
    assert np.array_equal(indices,stage_indices(stages,np.random.default_rng(4),size=3000))


def test_unavailable_dynamic_bank_keeps_normal_start(tmp_path):
    env = PlanetaryExplorationEnv(curriculum=False,difficulty_level=7,history_steps=4,
        training_stage_bank=str(tmp_path / "scenario_{level:02d}.json"),training_stage_probability=1)
    try:
        _,info = env.reset(seed=1)
        assert info["training_stage"] == 0
        assert env.steps == 0 and not env.inventory
    finally:
        env.close()


def test_dynamic_bank_does_not_reuse_previous_scenario_state(tmp_path):
    from pathlib import Path
    import shutil
    shutil.copy2("configs/training/level6-stages.json",tmp_path / "scenario_06.json")
    env = PlanetaryExplorationEnv(curriculum=False,difficulty_level=6,history_steps=4,
        training_stage_bank=str(tmp_path / "scenario_{level:02d}.json"),training_stage_probability=1)
    try:
        env.reset(seed=1)
        assert env.training_stage > 0
        env.reset(seed=1,options={"difficulty_level":7})
        assert env.training_stage == 0 and env.steps == 0 and not env.inventory
    finally:
        env.close()


import json
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pipelines.accelerated_curriculum import AcceleratedCurriculumCallback
from pipelines.specialist_policy import specialize_by_scenario, freeze_mastered
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum


@pytest.mark.parametrize("failure", [None, "short_mastery", "retention"])
def test_promotion_requires_full_mission_and_retention_gates(tmp_path, monkeypatch, failure):
    torch.set_num_threads(1)
    env = VecNormalize(DummyVecEnv([lambda: PlanetaryExplorationEnv(curriculum=False,history_steps=4,record_episodes=False)]),norm_reward=False)
    env.training = False
    model = PPO("MlpPolicy",env,n_steps=8,batch_size=8,device="cpu")
    specialize_by_scenario(model,env)
    parent = tmp_path / "parent"; parent.mkdir()
    (parent / "acceleration-history.json").write_text(json.dumps({"8":{"attempts":2}}))
    output = tmp_path / "run"; output.mkdir()
    manager = CampaignCurriculum(output / "curriculum_state.json",initial_level=7)
    manager.mastered = [1,2,3,4,5,6]; manager._save()
    config = {"accelerated_curriculum":{"max_updates":0},
              "environment":{"curriculum_state":str(output / "curriculum_state.json")}}
    protected = []
    class Guard:
        def protect_mastery(self, levels):
            protected.append(list(levels));freeze_mastered(model,levels)
    callback = AcceleratedCurriculumCallback(config,output,parent,Guard())
    callback.init_callback(model)
    monkeypatch.setattr(callback,"demonstration",lambda level:{
        "observations":np.zeros((4,132),dtype=np.float32),
        "actions":np.zeros(4,dtype=np.int64),"stages":np.arange(4)})
    calls=[]
    def evaluate(level,count,phase,seed):
        calls.append((level,count,phase))
        actual_count = 3 if failure == "short_mastery" and phase == "mastery" else count
        success = not (failure == "retention" and phase == "retention" and level == 3)
        return {"scenario_level":level,"evaluation_kind":"mastery",
                "cases":[{"seed":seed+i,"difficulty_level":level,"success":success,
                    "collisions":0,"samples_collected":4,"requested_samples":4,"battery":40} for i in range(actual_count)],
                "metrics":{"success_rate":float(success),"samples_collected_mean":4,
                           "collisions_per_episode":0,"steps_mean":100}}
    monkeypatch.setattr(callback,"evaluate",evaluate)
    try:
        callback._on_training_start()
        manager._load()
        if failure is None:
            assert manager.current_level == 8 and manager.mastered == list(range(1,8))
            assert protected == [list(range(1,8))]
            assert all((level,50,"retention") in calls for level in range(1,7))
            assert (output / "mastered-scenario-07-0/ppo_model.zip").exists()
        else:
            assert manager.current_level == 7 and manager.mastered == list(range(1,7))
            assert not protected
            assert not list(output.glob("mastered-*"))
    finally:
        env.close()


def test_stop_interrupts_supervised_work(tmp_path,monkeypatch):
    stop = tmp_path / "stop";stop.touch()
    monkeypatch.setenv("EXPEDITION_STOP_FILE",str(stop))
    callback = AcceleratedCurriculumCallback({"accelerated_curriculum":{}},tmp_path,None,None)
    with pytest.raises(KeyboardInterrupt):
        callback.check_stop()
