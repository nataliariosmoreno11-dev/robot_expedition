import numpy as np

from robot_expedition.planetary.qlearning import PlanetaryQLearningAgent, discretize_planetary


def observation():
    obs = np.ones(33, dtype=np.float32)
    obs[:9] = np.linspace(0.2, 5.0, 9)
    obs[10], obs[11] = 0.4, 0.8
    obs[14], obs[15], obs[21], obs[27], obs[32] = 0.8, 0.0, 1.0, 0.7, 0.5
    obs[18:21] = [1, 0, 0]
    obs[23:26] = [1, 0, 0]
    return obs


def test_planetary_qlearning_updates_and_restores_memory(tmp_path):
    agent = PlanetaryQLearningAgent(seed=4, epsilon=0.0)
    obs = observation()
    state, action = agent.choose(obs, guide_action=2)
    assert action == 2 and state == discretize_planetary(obs)
    agent.learn(state, action, 8.0, obs, True)
    agent.finish_episode()
    checkpoint = agent.save(tmp_path / "planetary-q.npz")

    restored = PlanetaryQLearningAgent(seed=9).load(checkpoint)
    assert restored.episodes == 1
    assert restored.updates == 1
    assert restored.snapshot()["states"] >= 1
    assert restored.values(state)[action] > 0


def test_planetary_qlearning_explores_all_valid_actions():
    agent = PlanetaryQLearningAgent(seed=2, epsilon=1.0)
    actions = {agent.choose(observation(), guide_action=0)[1] for _ in range(100)}
    assert actions == set(range(6))
