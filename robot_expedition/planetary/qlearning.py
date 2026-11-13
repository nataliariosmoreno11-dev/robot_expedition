"""Q-learning tabular disperso para el entorno planetario de 33 observaciones."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_SIZE = 11
ACTION_COUNT = 6


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return int(np.digitize(float(value), edges))


def discretize_planetary(observation) -> tuple[int, ...]:
    """Reduce sensores continuos a un estado compacto y generalizable."""
    obs = np.asarray(observation, dtype=np.float32)
    rays = obs[:9]
    angle = float(np.arctan2(obs[10], obs[11]))
    target_type = int(np.argmax(obs[18:21]))
    visible_type = int(np.argmax(obs[23:26])) if obs[21] > 0.5 else 3
    return (
        _bin(np.min(rays[3:6]), (0.45, 0.8, 1.5, 3.0)),
        _bin(np.mean(rays[:4]), (0.7, 1.5, 3.0)),
        _bin(np.mean(rays[5:]), (0.7, 1.5, 3.0)),
        _bin(angle, (-2.35, -0.8, -0.22, 0.22, 0.8, 2.35)),
        _bin(obs[9], (0.62, 1.4, 3.0, 5.0)),
        _bin(obs[14], (0.2, 0.45, 0.7)),
        _bin(obs[15], (0.01, 0.5, 0.99)),
        int(obs[21] > 0.5),
        int(target_type == visible_type),
        _bin(obs[27], (0.55, 0.72, 0.9)),
        _bin(obs[32], (0.25, 0.5, 0.75)),
    )


class PlanetaryQLearningAgent:
    """Tabla Q dispersa, epsilon-greedy y checkpoint comprimido reanudable."""

    def __init__(
        self,
        *,
        seed: int = 7,
        alpha: float = 0.18,
        gamma: float = 0.96,
        epsilon: float = 0.75,
        epsilon_min: float = 0.06,
        epsilon_decay: float = 0.992,
    ):
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.episodes = 0
        self.updates = 0
        self.q: dict[tuple[int, ...], np.ndarray] = {}

    def values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(ACTION_COUNT, dtype=np.float32)
        return self.q[state]

    def choose(self, observation, *, guide_action: int) -> tuple[tuple[int, ...], int]:
        state = discretize_planetary(observation)
        values = self.values(state)
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(ACTION_COUNT))
        elif not np.any(values):
            action = int(guide_action)
        else:
            action = int(np.argmax(values))
        return state, action

    def learn(self, state, action: int, reward: float, next_observation, done: bool) -> None:
        current = self.values(state)
        next_state = discretize_planetary(next_observation)
        future = 0.0 if done else float(np.max(self.values(next_state)))
        target = float(reward) + self.gamma * future
        current[action] += self.alpha * (target - float(current[action]))
        self.updates += 1

    def finish_episode(self) -> None:
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def snapshot(self) -> dict:
        return {
            "algorithm": "qlearning",
            "episodes": self.episodes,
            "updates": self.updates,
            "epsilon": self.epsilon,
            "states": len(self.q),
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        states = np.asarray(list(self.q), dtype=np.int16).reshape((-1, STATE_SIZE))
        values = np.asarray([self.q[state] for state in self.q], dtype=np.float32).reshape((-1, ACTION_COUNT))
        metadata = json.dumps(self.snapshot(), separators=(",", ":"))
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, states=states, q=values, metadata=np.asarray(metadata))
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> "PlanetaryQLearningAgent":
        source = Path(path)
        if not source.exists():
            return self
        with np.load(source, allow_pickle=False) as payload:
            states = payload["states"]
            values = payload["q"]
            metadata = json.loads(str(payload["metadata"].item()))
        if states.ndim != 2 or states.shape[1] != STATE_SIZE or values.shape != (len(states), ACTION_COUNT):
            raise ValueError("checkpoint Q-learning planetario incompatible")
        self.q = {tuple(int(item) for item in state): value.astype(np.float32) for state, value in zip(states, values)}
        self.episodes = int(metadata.get("episodes", 0))
        self.updates = int(metadata.get("updates", 0))
        self.epsilon = float(metadata.get("epsilon", self.epsilon))
        self.alpha = float(metadata.get("alpha", self.alpha))
        self.gamma = float(metadata.get("gamma", self.gamma))
        return self
