import pickle
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pipelines.specialist_policy import specialize, SpecialistPolicy


class FakePlanet(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (132,), dtype=np.float32)
    action_space = gym.spaces.Discrete(7)
    def reset(self, *, seed=None, options=None):
        return np.zeros(132, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(132, dtype=np.float32), 0.0, False, False, {}


def fixture():
    env = VecNormalize(DummyVecEnv([FakePlanet]), norm_reward=False)
    env.training = False
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=4, device="cpu")
    observations = np.random.default_rng(4).normal(size=(30,132)).astype(np.float32)
    observations[:, -1] = np.tile(np.arange(1,11)/10, 3)
    return model, env, observations


def test_specialization_preserves_logits_and_freezes_mastered_actor(tmp_path):
    torch.set_num_threads(1)
    model, env, observations = fixture()
    x = torch.tensor(env.normalize_obs(observations))
    before = model.policy.get_distribution(x).distribution.probs.detach().clone()
    assert specialize(model, env)
    assert torch.allclose(before, model.policy.get_distribution(x).distribution.probs, atol=1e-7)
    actor = model.policy.mlp_extractor.policy_net
    for _ in range(5):
        loss = -model.policy.get_distribution(x).log_prob(torch.zeros(30, dtype=torch.long)).mean()
        model.policy.optimizer.zero_grad(); loss.backward(); model.policy.optimizer.step()
    after = model.policy.get_distribution(x).distribution.probs.detach()
    old = observations[:, -1] <= 0.5
    assert torch.equal(before[old], after[old])
    assert not torch.equal(before[~old], after[~old])
    assert all(p.grad is None for p in actor.frozen.parameters())
    model.save(tmp_path / "candidate")
    loaded = PPO.load(tmp_path / "candidate.zip", env=env)
    assert isinstance(loaded.policy, SpecialistPolicy)
    assert torch.equal(after, loaded.policy.get_distribution(x).distribution.probs)
    assert not specialize(loaded, env)
    # Exercise actual PPO optimizer/rollout integration after reloading.
    loaded.learn(total_timesteps=8)
    assert torch.equal(before[old], loaded.policy.get_distribution(x).distribution.probs.detach()[old])
    env.close()


def test_specialist_requires_frozen_normalization():
    model, env, _ = fixture()
    env.training = True
    with pytest.raises(ValueError, match="frozen"):
        specialize(model, env)
    env.close()


def test_per_scenario_updates_and_mastery_survive_save_load(tmp_path):
    from pipelines.specialist_policy import specialize_by_scenario, freeze_mastered, ScenarioPolicy
    torch.set_num_threads(1)
    model, env, observations = fixture()
    x = torch.tensor(env.normalize_obs(observations))
    before = model.policy.get_distribution(x).distribution.probs.detach().clone()
    assert specialize_by_scenario(model, env)
    assert torch.allclose(before, model.policy.get_distribution(x).distribution.probs, atol=1e-7)
    # Grouping batches may change floating-point rounding at conversion.
    before = model.policy.get_distribution(x).distribution.probs.detach().clone()
    # Only train level six; all other actors must remain identical.
    six = observations[:, -1] == np.float32(.6)
    for _ in range(5):
        loss = -model.policy.get_distribution(x[six]).log_prob(torch.zeros(int(six.sum()), dtype=torch.long)).mean()
        model.policy.optimizer.zero_grad(); loss.backward(); model.policy.optimizer.step()
    after = model.policy.get_distribution(x).distribution.probs.detach()
    assert torch.equal(before[~six], after[~six])
    assert not torch.equal(before[six], after[six])
    freeze_mastered(model, [1,2,3,4,5,6])
    model.save(tmp_path / "scenario")
    loaded = PPO.load(tmp_path / "scenario.zip", env=env)
    assert isinstance(loaded.policy, ScenarioPolicy)
    assert loaded.policy.mlp_extractor.policy_net.frozen_levels[:6].all()
    assert torch.equal(after, loaded.policy.get_distribution(x).distribution.probs)
    # Critic-only rollouts in mastered scenarios still support backward().
    loaded.learn(total_timesteps=8)
    assert torch.equal(after, loaded.policy.get_distribution(x).distribution.probs)
    env.close()
