"""Inference wrapper saved as MLflow model code rather than a pickled object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class ExpeditionPolicy(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        config: dict[str, Any] = json.loads(
            Path(context.artifacts["config"]).read_text(encoding="utf-8")
        )
        self.algorithm = config["algorithm"]
        environment = config["environment"]

        if self.algorithm == "qlearning":
            from robot_expedition.envs.beacon_expedition import BeaconExpeditionEnv
            from robot_expedition.qlearning import QLearningAgent

            env = BeaconExpeditionEnv(level=config["level_id"], **environment)
            self.agent = QLearningAgent(env.action_space.n).load(context.artifacts["weights"])
            env.close()
            return

        def make_env():
            options = dict(environment)
            kind = options.pop("kind", "beacon")
            if kind == "planetary":
                from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv

                options.update(record_episodes=False, update_curriculum=False)
                env = PlanetaryExplorationEnv(**options)
                return Monitor(env, info_keywords=("success", "samples_collected", "difficulty_level"))
            from robot_expedition.envs.beacon_expedition import BeaconExpeditionEnv

            return Monitor(BeaconExpeditionEnv(level=config["level_id"], **options), info_keywords=("success",))

        base_env = DummyVecEnv([make_env])
        self.vec_env = VecNormalize.load(context.artifacts["vecnormalize"], base_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        self.model = PPO.load(context.artifacts["weights"], env=self.vec_env, device="auto")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: list[list[float]],
    ) -> list[int]:
        del context
        observations = np.asarray(model_input, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if self.algorithm == "qlearning":
            from robot_expedition.qlearning import discretize

            return [
                self.agent.choose(discretize(obs), explore=False, deterministic=True)
                for obs in observations
            ]
        normalized = self.vec_env.normalize_obs(observations)
        actions, _ = self.model.predict(normalized, deterministic=True)
        return np.asarray(actions).reshape(-1).astype(int).tolist()


mlflow.models.set_model(ExpeditionPolicy())
