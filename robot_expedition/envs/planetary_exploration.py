"""Ambiente MuJoCo para exploración marciana y recolección explícita.

Este contrato es optativo. ``BeaconExpeditionEnv`` conserva 20 observaciones y
cinco acciones para que los modelos históricos sigan siendo compatibles.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot_expedition.planetary.curriculum import CurriculumManager
from robot_expedition.planetary.campaign import FixedScenarioGenerator, catalog_hash
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum
from robot_expedition.planetary.episode_store import EpisodeStore, EpisodeWriter
from robot_expedition.planetary.scenario import (
    SAMPLE_TYPES,
    PlanetaryScenario,
    ScenarioGenerator,
)
from robot_expedition.planetary import curriculum as curriculum_module
from robot_expedition.planetary import profiles as profiles_module
from robot_expedition.planetary import scenario as scenario_module

from .beacon_expedition import (
    ACTIONS,
    BASE,
    CRACK,
    DT,
    RAY_RANGE,
    SAND,
    SUBSTEPS,
    V_MAX,
    W_MAX,
    WHEEL_BASE,
    WHEEL_R,
    BeaconExpeditionEnv,
    _in_rect,
    _wrap,
)


PLANETARY_ENVIRONMENT_VERSION = "planetary-exploration/2.0.0"
COLLECT_ACTION = len(ACTIONS)
PLANETARY_ACTIONS = [*ACTIONS, ("recolectar", 0.0, 0.0), ("retroceder", -0.35, 0.0)]
COLLECTION_RADIUS = 0.68
OBSERVATION_SIZE = 33


def planetary_environment_hash(environment: dict | None = None) -> str:
    """Hash del ambiente, generador, perfiles, currículo y configuración."""
    configuration = dict(environment or {})
    for operational_key in (
        "episode_db",
        "curriculum_state",
        "run_id",
        "policy_version",
        "mode",
        "record_episodes",
        "max_experiments",
        "update_curriculum",
    ):
        configuration.pop(operational_key, None)
    payload = "".join(
        inspect.getsource(module)
        for module in (scenario_module, profiles_module, curriculum_module)
    )
    payload += inspect.getsource(PlanetaryExplorationEnv)
    payload += catalog_hash()
    from robot_expedition.planetary import campaign, campaign_curriculum, reference
    from . import beacon_expedition

    payload += "".join(
        inspect.getsource(m)
        for m in (campaign, campaign_curriculum, reference, beacon_expedition)
    )
    payload += json.dumps(
        configuration, sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PlanetaryExplorationEnv(BeaconExpeditionEnv):
    """Rover físico con misión procedural de recolección de muestras."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        *,
        planet: str = "mars",
        difficulty_level: int = 1,
        curriculum: bool = True,
        curriculum_state: str | Path | None = None,
        curriculum_settings: dict | None = None,
        episode_db: str | Path | None = None,
        record_episodes: bool = False,
        max_experiments: int | None = None,
        campaign: bool = True,
        history_steps: int = 1,
        run_id: str = "interactive",
        policy_version: str = "unversioned",
        mode: str = "entrenamiento",
        max_steps: int | None = None,
        render_mode: str | None = None,
        update_curriculum: bool = True,
    ):
        self.campaign = campaign
        if campaign and planet != "mars":
            raise ValueError("the fixed campaign currently defines Mars missions only")
        self.history_steps = int(history_steps)
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        self._history = []
        self.generator = (
            FixedScenarioGenerator() if campaign else ScenarioGenerator(planet)
        )
        self.requested_level = int(difficulty_level)
        self.curriculum_enabled = bool(curriculum)
        self.curriculum = (CampaignCurriculum if campaign else CurriculumManager)(
            curriculum_state,
            initial_level=difficulty_level,
            **(curriculum_settings or {}),
        )
        self.update_curriculum = bool(update_curriculum)
        self.configured_max_steps = max_steps
        self.run_id = str(run_id)
        self.policy_version = str(policy_version)
        self.mode = str(mode)
        self.store = (
            EpisodeStore(
                episode_db, max_experiments=None if campaign else max_experiments
            )
            if record_episodes and episode_db
            else None
        )
        self.writer: EpisodeWriter | None = None
        self.episode_started = 0.0
        self.total_reward = 0.0
        self.scenario: PlanetaryScenario | None = None
        self.episode_seed = 0
        self.samples: list[dict] = []
        self.collected_ids: set[str] = set()
        self.inventory: list[str] = []
        self.target_type = SAMPLE_TYPES[0]
        self.requested_samples = 1
        self.capacity = 1
        self.visibility_range = RAY_RANGE
        self._last_event: str | None = None
        self._stuck_steps = 0
        super().__init__(
            level="n2",
            max_steps=max_steps or 540,
            ray_noise=0.0,
            render_mode=render_mode,
        )
        self.action_space = spaces.Discrete(len(PLANETARY_ACTIONS))
        low = np.array(
            [0.0] * 9
            + [0.0, -1.0, -1.0, -1.0, -1.0, 0.0]
            + [0.0, 0.0, 0.0]
            + [0.0] * 3
            + [0.0, 0.0]
            + [0.0] * 3
            + [0.0, 0.0, 0.0, 0.0, -math.pi, -math.pi, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [RAY_RANGE] * 9
            + [17.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            + [1.0, 1.0, 1.0]
            + [1.0] * 3
            + [1.0, 17.0]
            + [1.0] * 3
            + [1.0, 1.0, 1.0, 1.0, math.pi, math.pi, 1.0],
            dtype=np.float32,
        )
        if len(low) != OBSERVATION_SIZE:
            raise AssertionError(
                f"contrato planetario inválido: {len(low)} observaciones"
            )
        self.observation_space = spaces.Box(
            np.tile(low, self.history_steps),
            np.tile(high, self.history_steps),
            dtype=np.float32,
        )
        self._terrain_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
        )

    # --------------------------------------------------------- escenario
    def _finish_interrupted(self) -> None:
        if self.writer is not None and not self.writer.closed:
            self.writer.finish(
                end="reinicio",
                success=False,
                cumulative_reward=self.total_reward,
                duration_s=max(0.0, time.perf_counter() - self.episode_started),
                steps=self.steps,
                samples_collected=len(self.inventory),
                collisions=self.collisions,
                path_length=self.path_length,
            )

    def _apply_terrain(self, scenario: PlanetaryScenario) -> None:
        self.heights = scenario.terrain_heights.copy()
        lo = float(self.heights.min())
        elevation = max(0.01, float(self.heights.max() - lo))
        self._terrain_lo = lo
        self.model.hfield_size[0, 2] = elevation
        self.model.geom_pos[self._terrain_id, 2] = lo
        normalized = np.clip((self.heights - lo) / elevation, 0.0, 1.0)
        self.model.hfield_data[:] = normalized.ravel()
        self.model.geom_friction[self._terrain_id] = [scenario.traction, 0.02, 0.002]

    def _apply_rocks(self, rocks: list[tuple]) -> None:
        if len(rocks) != len(self._rock_ids):
            raise ValueError("physical obstacle count mismatch")
        self.rocks = list(rocks)
        for gid in self._rock_ids:
            self.model.geom_pos[gid, 2] = -5.0
        for gid, (cx, cy, hx, hy, height) in zip(self._rock_ids, rocks):
            self.model.geom_size[gid] = [hx, hy, height / 2.0]
            self.model.geom_pos[gid] = [cx, cy, self.ground(cx, cy) + height / 2.0]

    def _compile_scenario(self, scenario):
        # Static obstacle broad-phase bounds are compiled by MuJoCo. Resizing
        # placeholder geoms alone leaves stale bounds and permits tunnelling.
        if getattr(self, "_compiled_scenario", None) == scenario.scenario_id:
            return
        from .beacon_expedition import _build_xml

        rocks = [(f"R{i}", *r) for i, r in enumerate(scenario.rocks)]
        xml, _, _, _ = _build_xml(rocks=rocks, dunes=[])
        self.model = mujoco.MjModel.from_xml_string(xml)
        # 5 ms contact integration prevents artificial energy spikes when small
        # wheels cross the heightfield. Control decisions remain at 2 Hz.
        self.model.opt.timestep = 0.005
        self._substeps = round(DT / self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)
        gid = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self._rover = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        self._sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "sensor")
        self._rock_ids = [gid(f"rock_{i}") for i in range(len(rocks))]
        self._rock_names = [r[0] for r in rocks]
        self._block_id = gid("block")
        self._terrain_id = gid("terrain")
        self._obstacle_ids = (
            set(self._rock_ids)
            | {self._block_id}
            | {gid(w) for w in ("wall_n", "wall_s", "wall_w", "wall_e")}
        )
        self._is_obstacle = np.isin(
            np.arange(self.model.ngeom), list(self._obstacle_ids)
        )
        self._is_rover = self.model.geom_bodyid != 0
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._compiled_scenario = scenario.scenario_id

    def reset(self, *, seed=None, options=None):
        self._finish_interrupted()
        self._history = []
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if options.get("scenario_id"):
            from robot_expedition.planetary.campaign import SCENARIO_IDS

            if options["scenario_id"] not in SCENARIO_IDS:
                raise ValueError("unknown scenario_id")
            options["difficulty_level"] = SCENARIO_IDS.index(options["scenario_id"]) + 1
        level = int(
            options.get("difficulty_level")
            or (
                self.curriculum.select_level(self.np_random)
                if self.curriculum_enabled
                else self.requested_level
            )
        )
        scenario_seed = (
            int(options.get("scenario_seed", seed))
            if options.get("scenario_seed", seed) is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        scenario = self.generator.generate(scenario_seed, level)
        self._compile_scenario(scenario)
        self.scenario = scenario
        self.episode_seed = scenario_seed
        self.scenario_id = scenario.scenario_id
        self.scenario_difficulty = f"L{level} · {scenario.difficulty_name}"
        self.model.opt.gravity[:] = [0.0, 0.0, -scenario.planet["gravity_m_s2"]]
        self._apply_terrain(scenario)
        self._apply_rocks(scenario.rocks)

        self.samples = [
            {**sample.to_dict(), "collected": False} for sample in scenario.samples
        ]
        self.target_type = scenario.target_type
        self.requested_samples = scenario.requested_samples
        self.capacity = scenario.capacity
        self.collected_ids = set()
        self.inventory = []
        self.beacons = [
            (item["x"], item["y"]) for item in self.samples if item["target"]
        ]
        self.visibility_range = max(
            2.0, RAY_RANGE * (1.0 - 0.58 * scenario.dust_opacity)
        )
        self.max_steps = int(self.configured_max_steps or scenario.max_steps)

        mujoco.mj_resetData(self.model, self.data)
        self._set_block(False)
        start = list(scenario.start)
        if options.get("robustness"):
            start[0] += float(self.np_random.uniform(-0.12, 0.12))
            start[1] += float(self.np_random.uniform(-0.12, 0.12))
            start[2] += float(self.np_random.uniform(-0.2, 0.2))
        self.set_pose(*start)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)
        self.initial_position = [self.x, self.y]
        self.collection_positions = []
        self.battery = scenario.battery_budget
        self.stage = 1
        self.visited = [False] * self.requested_samples
        self.steps = self.collisions = 0
        self.path_length = 0.0
        self.route_cost = 0.0
        self.explored_cells = set()
        self._done = False
        self.total_reward = 0.0
        self._last_event = None
        self._stuck_steps = 0
        self.episode_started = time.perf_counter()
        observation = self._observation()

        self.writer = None
        if self.store is not None:
            self.writer = self.store.start_episode(
                run_id=self.run_id,
                policy_version=self.policy_version,
                mode=self.mode,
                seed=scenario_seed,
                scenario=scenario.to_dict(),
                map_spec=self.map_spec(),
            )
            self.writer.record_step(
                state=self._snapshot(),
                action=-1,
                action_name="inicio",
                reward=0.0,
                cumulative_reward=0.0,
                event="inicio",
                collision=False,
                terminated=False,
                truncated=False,
            )
        return observation, self._info(None)

    # --------------------------------------------------------- sensores
    def _remaining_targets(self) -> list[dict]:
        return [
            sample
            for sample in self.samples
            if sample["target"] and not sample["collected"]
        ]

    def target(self):
        if len(self.inventory) >= self.requested_samples:
            return BASE[:2]
        remaining = self._remaining_targets()
        if not remaining:
            return BASE[:2]
        sample = min(
            remaining,
            key=lambda item: math.hypot(item["x"] - self.x, item["y"] - self.y),
        )
        return sample["x"], sample["y"]

    def rays(self):
        rays = super().rays()
        if self.scenario is None:
            return rays
        noise = 0.015 + self.scenario.dust_opacity * 0.16
        rays = np.minimum(rays, self.visibility_range)
        rays += self.np_random.normal(0.0, noise, len(rays)).astype(np.float32)
        return np.clip(rays, 0.0, self.visibility_range)

    def _nearest_visible_sample(self) -> tuple[dict | None, float]:
        available = [
            sample
            for sample in self.samples
            if not sample["collected"] and self._sample_is_visible(sample)
        ]
        if not available:
            return None, 17.0
        sample = min(
            available,
            key=lambda item: math.hypot(item["x"] - self.x, item["y"] - self.y),
        )
        distance = math.hypot(sample["x"] - self.x, sample["y"] - self.y)
        return sample, distance

    def _sample_is_visible(self, sample: dict) -> bool:
        distance = math.hypot(sample["x"] - self.x, sample["y"] - self.y)
        if distance > self.visibility_range:
            return False
        # Detector visual simplificado con oclusión por rocas; no revela muestras
        # detrás de obstáculos ni fuera del alcance reducido por el polvo.
        points = max(2, int(distance / 0.16))
        for fraction in np.linspace(0.05, 0.95, points):
            x = self.x + (sample["x"] - self.x) * float(fraction)
            y = self.y + (sample["y"] - self.y) * float(fraction)
            if any(_in_rect(x, y, rock, margin=0.08) for rock in self.rocks):
                return False
        return True

    def _local_traction(self, x: float | None = None, y: float | None = None) -> float:
        x = self.x if x is None else float(x)
        y = self.y if y is None else float(y)
        traction = self.scenario.traction
        for cx, cy, half_x, half_y, coefficient in self.scenario.traction_zones:
            if abs(x - cx) <= half_x and abs(y - cy) <= half_y:
                traction = min(traction, coefficient)
        return float(traction)

    def _observation(self):
        self.last_rays = self.rays()
        speed = max(-1.0, min(1.0, self._forward_speed() / V_MAX))
        turn = max(-1.0, min(1.0, float(self.data.qvel[5]) / W_MAX))
        collected = min(len(self.inventory), self.requested_samples)
        requested = max(1, self.requested_samples)
        target_one_hot = [
            float(
                any(
                    s["target"] and not s["collected"] and s["sample_type"] == item
                    for s in self.samples
                )
            )
            for item in SAMPLE_TYPES
        ]
        visible, visible_distance = self._nearest_visible_sample()
        visible_one_hot = [
            float(visible is not None and visible["sample_type"] == item)
            for item in SAMPLE_TYPES
        ]
        if collected >= requested:
            tx, ty = BASE[:2]
            navigation_distance = math.hypot(tx - self.x, ty - self.y)
            relative = _wrap(math.atan2(ty - self.y, tx - self.x) - self.theta)
            relative_sin, relative_cos = math.sin(relative), math.cos(relative)
        else:
            visible_target = (
                visible if visible is not None and visible["target"] else None
            )
            if visible_target is None:
                navigation_distance = RAY_RANGE
                relative_sin = relative_cos = 0.0
            else:
                navigation_distance = visible_distance
                relative = _wrap(
                    math.atan2(
                        visible_target["y"] - self.y, visible_target["x"] - self.x
                    )
                    - self.theta
                )
                relative_sin, relative_cos = math.sin(relative), math.cos(relative)
        scenario = self.scenario
        gravity = scenario.planet["gravity_m_s2"] / 9.81 if scenario else 0.0
        traction = self._local_traction() if scenario else 0.0
        wind = min(1.0, scenario.wind_speed_m_s / 30.0) if scenario else 0.0
        level = (
            scenario.difficulty_level / (10.0 if self.campaign else 8.0)
            if scenario
            else 0.0
        )
        pitch, roll = self.tilt
        tail = [
            navigation_distance,
            relative_sin,
            relative_cos,
            speed,
            turn,
            self.battery / 100.0,
            collected / requested,
            (requested - collected) / requested,
            len(self.inventory) / max(1, self.capacity),
            *target_one_hot,
            float(visible is not None),
            visible_distance,
            *visible_one_hot,
            gravity,
            traction,
            self.visibility_range / RAY_RANGE,
            wind,
            pitch,
            roll,
            level,
        ]
        observation = np.concatenate(
            [self.last_rays, np.asarray(tail, dtype=np.float32)]
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise AssertionError(
                f"observación planetaria inesperada: {observation.shape}"
            )
        self._history.append(observation)
        self._history = self._history[-self.history_steps :]
        while len(self._history) < self.history_steps:
            self._history.insert(0, observation.copy())
        self.last_observation = np.concatenate(self._history)
        return self.last_observation.copy()

    # --------------------------------------------------------- dinámica
    def _wind_force(self) -> tuple[float, float]:
        scenario = self.scenario
        if scenario is None:
            return 0.0, 0.0
        # La ráfaga varía, pero su fuerza usa la baja densidad real aproximada de Marte.
        gust = 1.0 + 0.28 * math.sin(self.steps * DT / 7.0 + scenario.seed % 13)
        speed = scenario.wind_speed_m_s * gust
        force = (
            0.5
            * scenario.planet["atmosphere_density_kg_m3"]
            * scenario.planet["wind_drag_coefficient"]
            * scenario.planet["rover_frontal_area_m2"]
            * speed**2
        )
        direction = scenario.wind_direction_rad + 0.15 * math.sin(
            self.steps * DT / 11.0
        )
        return force * math.cos(direction), force * math.sin(direction)

    def _collect(self) -> tuple[float, str]:
        nearby = [
            sample
            for sample in self.samples
            if math.hypot(sample["x"] - self.x, sample["y"] - self.y)
            <= COLLECTION_RADIUS
        ]
        if not nearby:
            if len(self.inventory) >= self.requested_samples and _in_rect(
                self.x, self.y, BASE
            ):
                return 0.0, "inventario completo; regreso confirmado"
            return -0.6, "recolección sin muestra"
        sample = min(
            nearby, key=lambda item: math.hypot(item["x"] - self.x, item["y"] - self.y)
        )
        if sample["collected"]:
            return -1.2, f"muestra {sample['sample_id']} ya recolectada"
        if not sample["target"]:
            return -2.5, f"muestra incorrecta: {sample['sample_type']}"
        if len(self.inventory) >= self.capacity:
            return -1.5, "inventario lleno"
        sample["collected"] = True
        self.collected_ids.add(sample["sample_id"])
        self.inventory.append(sample["sample_id"])
        self.collection_positions.append([self.x, self.y])
        self.visited[min(len(self.inventory) - 1, len(self.visited) - 1)] = True
        self.stage = (
            3
            if len(self.inventory) >= self.requested_samples
            else min(2, len(self.inventory) + 1)
        )
        bonus = 16.0 if len(self.inventory) == self.requested_samples else 12.0
        return bonus, f"recolectada {sample['sample_id']} ({sample['sample_type']})"

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required after episode termination")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"acción fuera de rango: {action}")
        _, velocity, angular = PLANETARY_ACTIONS[action]
        distance_before = self._distance_to_target()
        path_before = self.path_length
        self.data.ctrl[:] = [
            (velocity - angular * WHEEL_BASE / 2.0) / WHEEL_R,
            (velocity + angular * WHEEL_BASE / 2.0) / WHEEL_R,
        ]
        collision = False
        wind_x, wind_y = self._wind_force()
        for _ in range(SUBSTEPS):
            x0, y0 = self.x, self.y
            local_traction = self._local_traction(x0, y0)
            self.model.geom_friction[self._terrain_id] = [local_traction, 0.02, 0.002]
            sand_drag = (
                -12.0 * self.data.qvel[0:2] if _in_rect(x0, y0, SAND) else np.zeros(2)
            )
            rolling_multiplier = (
                1.0 + max(0.0, self.scenario.traction - local_traction) * 2.4
            )
            rolling = (
                -self.scenario.planet["rolling_resistance"]
                * rolling_multiplier
                * self.data.qvel[0:2]
            )
            self.data.xfrc_applied[self._rover, 0:2] = (
                sand_drag + rolling + [wind_x, wind_y]
            )
            mujoco.mj_step(self.model, self.data, nstep=self._substeps // SUBSTEPS)
            moved = math.hypot(self.x - x0, self.y - y0)
            self.path_length += moved
            if self.data.ncon:
                geoms = self.data.contact.geom
                obstacles, rover = self._is_obstacle, self._is_rover
                collision |= bool(
                    np.any(
                        (obstacles[geoms[:, 0]] & rover[geoms[:, 1]])
                        | (obstacles[geoms[:, 1]] & rover[geoms[:, 0]])
                    )
                )

        speed = self._forward_speed()
        turn = float(self.data.qvel[5])
        energy = min(1.0, (speed / V_MAX) ** 2 + 0.25 * (turn / W_MAX) ** 2)
        cold_factor = 1.0 + max(0.0, -40.0 - self.scenario.temperature_c) / 180.0
        traction_factor = 1.0 + max(0.0, 0.75 - self._local_traction()) * 0.8
        wind_factor = 1.0 + min(0.25, math.hypot(wind_x, wind_y) / 20.0)
        drain = (0.10 + 0.40 * energy) * cold_factor * traction_factor * wind_factor
        drain *= self.scenario.energy_scale
        self.battery = max(0.0, self.battery - drain)
        pitch, roll = self.tilt
        from robot_expedition.planetary.reference import terrain_density

        self.route_cost += (self.path_length - path_before) * terrain_density(
            self.scenario, self.x, self.y
        )
        self.steps += 1
        self.collisions += int(collision)

        progress = distance_before - self._distance_to_target()
        cell = (int(self.x / 0.5), int(self.y / 0.5))
        discovery = cell not in self.explored_cells
        self.explored_cells.add(cell)
        # No hidden-target distance shaping: a necessary detour costs no extra
        # directional penalty. One-time coverage cannot be farmed by loops.
        reward = 0.12 * discovery - 0.01 - 0.05 * drain - 0.6 * collision
        event = None
        if action == COLLECT_ACTION:
            collect_reward, event = self._collect()
            reward += collect_reward

        if action < COLLECT_ACTION and abs(speed) < 0.025:
            self._stuck_steps += 1
        else:
            self._stuck_steps = max(0, self._stuck_steps - 1)
        if self._stuck_steps >= 12:
            reward -= 0.12
            event = event or "rover atascado"

        enough_samples = len(self.inventory) >= self.requested_samples
        success = (
            enough_samples and _in_rect(self.x, self.y, BASE) and self.battery > 0.0
        )
        end = None
        if success:
            reward += 250.0 + 15.0 * self.battery / 100.0
            end = "exito"
            event = "misión completada"
        elif _in_rect(self.x, self.y, CRACK) or self.z < -0.15:
            reward -= 50.0
            end = "grieta"
        elif self.upright < 0.3:
            reward -= 50.0
            end = "volco"
        elif self.battery <= 0.0:
            reward -= 50.0
            end = "bateria"
        elif (
            self.x > self.scenario.active_x_max + 0.2
            or abs(self.y) > self.scenario.active_y_half + 0.2
        ):
            reward -= 35.0
            end = "fuera_area"
        terminated = end is not None
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            end = "tiempo"

        self._done = terminated or truncated
        self.total_reward += float(reward)
        self._last_event = event
        observation = self._observation()
        info = self._info(end)
        info.update(collision=collision, event=event, energy=energy, progress=progress)
        if self.writer is not None:
            self.writer.record_step(
                state=self._snapshot(action, reward, collision),
                action=action,
                action_name=PLANETARY_ACTIONS[action][0],
                reward=reward,
                cumulative_reward=self.total_reward,
                event=event,
                collision=collision,
                terminated=terminated,
                truncated=truncated,
            )
        if terminated or truncated:
            if self.writer is not None:
                self.writer.finish(
                    end=end or "desconocido",
                    success=success,
                    cumulative_reward=self.total_reward,
                    duration_s=time.perf_counter() - self.episode_started,
                    steps=self.steps,
                    samples_collected=len(self.inventory),
                    collisions=self.collisions,
                    path_length=self.path_length,
                )
            if self.curriculum_enabled and self.update_curriculum:
                info["curriculum_advanced"] = self.curriculum.record_result(
                    success, evaluated_level=self.scenario.difficulty_level
                )
                info["curriculum"] = self.curriculum.snapshot()
        return observation, float(reward), terminated, truncated, info

    # --------------------------------------------------------- contratos
    def _info(self, end):
        requested = max(1, self.requested_samples)
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "theta": self.theta,
            "stage": self.stage,
            "visited": list(self.visited),
            "battery": self.battery,
            "collisions": self.collisions,
            "path_length": self.path_length,
            "route_cost": self.route_cost,
            "collection_positions": self.collection_positions,
            "initial_position": self.initial_position,
            "energy_consumed": self.scenario.battery_budget - self.battery,
            "success": end == "exito",
            "end": end,
            "scenario_id": self.scenario_id,
            "difficulty": self.scenario_difficulty,
            "difficulty_level": self.scenario.difficulty_level
            if self.scenario
            else self.requested_level,
            "planet": self.scenario.planet["planet_id"] if self.scenario else "mars",
            "target_type": self.target_type,
            "requested_samples": self.requested_samples,
            "samples_collected": len(self.inventory),
            "inventory": list(self.inventory),
            "mission_progress": (min(len(self.inventory), requested) + (end == "exito"))
            / (requested + 1),
        }

    def _frame(
        self, action: int = -1, reward: float = 0.0, collision: bool = False
    ) -> list:
        return [
            round(self.x, 3),
            round(self.y, 3),
            round(self.z, 3),
            *np.round(self.quat, 4).tolist(),
            self.stage,
            round(self.battery, 2),
            int(action),
            round(float(reward), 3),
            int(collision),
            [round(float(ray), 2) for ray in self.last_rays],
        ]

    def _snapshot(
        self, action: int = -1, reward: float = 0.0, collision: bool = False
    ) -> dict:
        return {
            "frame": self._frame(action, reward, collision),
            "observation": self.last_observation.tolist(),
            "policy_version": self.policy_version,
            "pose": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "quaternion": self.quat.tolist(),
                "theta": self.theta,
            },
            "mission": {
                "target_type": self.target_type,
                "requested": self.requested_samples,
                "collected": len(self.inventory),
                "inventory": list(self.inventory),
                "samples": [dict(item) for item in self.samples],
                "event": self._last_event,
                "status": "completada"
                if self._done
                and _in_rect(self.x, self.y, BASE)
                and len(self.inventory) >= self.requested_samples
                else (
                    "regresando"
                    if len(self.inventory) >= self.requested_samples
                    else "recolectando"
                ),
            },
            "resources": {
                "battery": self.battery,
                "energy_consumed": self.scenario.battery_budget - self.battery,
                "path_length": self.path_length,
                "route_cost": self.route_cost,
                "initial_position": self.initial_position,
                "collection_positions": self.collection_positions,
                "final_position": [self.x, self.y],
            },
            "environment": {
                "planet": self.scenario.planet["planet_id"],
                "gravity_m_s2": self.scenario.planet["gravity_m_s2"],
                "traction": self._local_traction(),
                "wind_speed_m_s": self.scenario.wind_speed_m_s,
                "dust_opacity": self.scenario.dust_opacity,
                "temperature_c": self.scenario.temperature_c,
                "visibility_m": self.visibility_range,
            },
        }

    def map_spec(self):
        spec = super().map_spec()
        if self.scenario is None:
            return spec
        spec.update(
            environment_version=PLANETARY_ENVIRONMENT_VERSION,
            scenario_id=self.scenario.scenario_id,
            difficulty=self.scenario_difficulty,
            difficulty_level=self.scenario.difficulty_level,
            active_extent=[
                self.scenario.active_x_max,
                self.scenario.active_y_half * 2.0,
            ],
            samples=[dict(item) for item in self.samples],
            mission={
                "target_type": self.target_type,
                "requested_samples": self.requested_samples,
                "capacity": self.capacity,
            },
            environment={
                "planet": self.scenario.planet,
                "traction": self.scenario.traction,
                "wind_speed_m_s": self.scenario.wind_speed_m_s,
                "wind_direction_rad": self.scenario.wind_direction_rad,
                "dust_opacity": self.scenario.dust_opacity,
                "temperature_c": self.scenario.temperature_c,
                "visibility_m": self.visibility_range,
                "dunes": [list(item) for item in self.scenario.dunes],
                "craters": [list(item) for item in self.scenario.craters],
                "traction_zones": [list(item) for item in self.scenario.traction_zones],
            },
            actions=[item[0] for item in PLANETARY_ACTIONS],
        )
        return spec

    def close(self):
        self._finish_interrupted()
        if self.store is not None:
            self.store.close()
            self.store = None
        super().close()
