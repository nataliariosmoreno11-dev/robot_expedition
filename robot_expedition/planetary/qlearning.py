"""Q-learning tabular disperso para el entorno planetario de 33 observaciones."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_SIZE = 11
ACTION_COUNT = 7
COLLECT_ACTION = 5


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
        alpha: float = 0.14,
        gamma: float = 0.98,
        epsilon: float = 0.75,
        epsilon_min: float = 0.06,
        epsilon_decay: float = 0.992,
        guide_probability: float = 0.95,
        warmup_visits: int = 6,
    ):
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.guide_probability = float(guide_probability)
        self.warmup_visits = int(warmup_visits)
        self.episodes = 0
        self.updates = 0
        self.q: dict[tuple[int, ...], np.ndarray] = {}
        self.visits: dict[tuple[int, ...], np.ndarray] = {}
        self.guided_decisions = 0
        self.random_decisions = 0
        self.greedy_decisions = 0

    def values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(ACTION_COUNT, dtype=np.float32)
        return self.q[state]

    def visit_counts(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.visits:
            self.visits[state] = np.zeros(ACTION_COUNT, dtype=np.int32)
        return self.visits[state]

    def choose(self, observation, *, guide_action: int) -> tuple[tuple[int, ...], int]:
        state = discretize_planetary(observation)
        values = self.values(state)
        visits = self.visit_counts(state)
        obs = np.asarray(observation, dtype=np.float32)
        front_blocked = float(np.min(obs[3:6])) < 0.9
        barely_moving = abs(float(obs[12])) < 0.035

        # Recolectar es una acción escasa y de alta recompensa. Cuando el
        # controlador sensorial confirma que procede, no se desperdicia ese
        # instante con una acción aleatoria.
        if int(guide_action) == COLLECT_ACTION:
            action = int(guide_action)
            self.guided_decisions += 1
        elif front_blocked or barely_moving:
            # Escudo de seguridad: la exploracion no puede ordenar avanzar
            # contra una roca ni anular una maniobra de desatasco.
            action = int(guide_action)
            self.guided_decisions += 1
        elif self.rng.random() < self.epsilon:
            # Exploración guiada: mantiene variedad, pero la mayor parte de
            # los pasos exploratorios siguen una maniobra segura. El esquema
            # anterior era 100 % aleatorio y hacía parecer que el rover no
            # aprendía incluso después de actualizar su tabla.
            if self.rng.random() < self.guide_probability:
                action = int(guide_action)
                self.guided_decisions += 1
            else:
                action = int(self.rng.integers(ACTION_COUNT))
                self.random_decisions += 1
        elif int(visits.sum()) < self.warmup_visits or not np.any(values):
            action = int(guide_action)
            self.guided_decisions += 1
        else:
            action = int(np.argmax(values))
            self.greedy_decisions += 1
        return state, action

    def learn(self, state, action: int, reward: float, next_observation, done: bool) -> None:
        current = self.values(state)
        self.visit_counts(state)[action] += 1
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
            "guide_probability": self.guide_probability,
            "guided_decisions": self.guided_decisions,
            "random_decisions": self.random_decisions,
            "greedy_decisions": self.greedy_decisions,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        states = np.asarray(list(self.q), dtype=np.int16).reshape((-1, STATE_SIZE))
        values = np.asarray([self.q[state] for state in self.q], dtype=np.float32).reshape((-1, ACTION_COUNT))
        visits = np.asarray([self.visit_counts(state) for state in self.q], dtype=np.int32).reshape((-1, ACTION_COUNT))
        metadata = json.dumps(self.snapshot(), separators=(",", ":"))
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, states=states, q=values, visits=visits, metadata=np.asarray(metadata))
        temporary.replace(destination)
        return destination

    def load(self, path: str | Path) -> "PlanetaryQLearningAgent":
        source = Path(path)
        if not source.exists():
            return self
        with np.load(source, allow_pickle=False) as payload:
            states = payload["states"]
            values = payload["q"]
            visits = payload["visits"] if "visits" in payload.files else np.zeros_like(values, dtype=np.int32)
            metadata = json.loads(str(payload["metadata"].item()))
        if (states.ndim != 2 or states.shape[1] != STATE_SIZE
                or values.shape != (len(states), ACTION_COUNT)
                or visits.shape != (len(states), ACTION_COUNT)):
            raise ValueError("checkpoint Q-learning planetario incompatible")
        self.q = {tuple(int(item) for item in state): value.astype(np.float32) for state, value in zip(states, values)}
        self.visits = {tuple(int(item) for item in state): value.astype(np.int32) for state, value in zip(states, visits)}
        self.episodes = int(metadata.get("episodes", 0))
        self.updates = int(metadata.get("updates", 0))
        self.epsilon = float(metadata.get("epsilon", self.epsilon))
        self.alpha = float(metadata.get("alpha", self.alpha))
        self.gamma = float(metadata.get("gamma", self.gamma))
        # Los checkpoints antiguos conservan la tabla aprendida, pero adoptan
        # el escudo y la exploracion guiada actuales.
        self.guide_probability = max(self.guide_probability, float(metadata.get("guide_probability", 0.0)))
        self.guided_decisions = int(metadata.get("guided_decisions", 0))
        self.random_decisions = int(metadata.get("random_decisions", 0))
        self.greedy_decisions = int(metadata.get("greedy_decisions", 0))
        return self
