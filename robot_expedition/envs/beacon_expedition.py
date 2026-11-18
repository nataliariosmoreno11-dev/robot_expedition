"""Ambiente 3D de Robot Expedition (ficha docs/ambiente-v0.1.md).

Un rover con traccion diferencial sale de la base, visita B1 y luego B2, y
vuelve a la base antes de agotar la bateria. La fisica es MuJoCo en 3D:
gravedad de Marte, terreno con dunas (heightfield), una grieta real en la que
puede caer, rocas y paredes con colision, ruedas con motor y rayos lanzados en
3D desde un sensor del rover.
"""

from __future__ import annotations

import math
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import yaml


ENVIRONMENT_VERSION = "robot-expedition-3d/1.1.0"
SCENARIOS_DIR = Path(__file__).parents[2] / "configs" / "scenarios"


# --- Mapa (metros). Rectangulos como (cx, cy, medio_ancho_x, medio_ancho_y, alto).
MAP_W, MAP_H = 12.0, 12.0                       # x en [0, 12], y en [-6, 6]
BASE = (1.5, 0.0, 1.0, 1.0)                     # centro y medio lado
START = (1.5, 0.0, 0.0)                         # x, y, rumbo (rad)
BEACONS = [(9.5, 3.5), (9.5, -3.5)]
VISIT_RADIUS = 0.6
ROCKS = [
    ("R1", 5.0, 1.5, 0.5, 0.8, 0.8),
    ("R2", 5.0, -2.5, 0.6, 0.5, 0.6),
    ("R3", 8.0, 0.0, 0.4, 1.0, 1.0),
    ("R4", 3.0, -4.5, 0.5, 0.5, 0.5),
    ("R5", 10.5, 5.0, 0.4, 0.4, 0.7),
]
BLOCK = ("bloqueo", 9.7, 0.0, 1.1, 0.3, 0.8)   # perturbacion: corredor bloqueado
CRACK = (3.5, 3.5, 0.5, 0.2)                    # hueco de 0.5 m de profundidad
CRACK_DEPTH = 0.5
SAND = (7.0, -4.5, 1.0, 0.7)                    # arena suelta: frena al rover
# Dunas: (cx, cy, altura m, sigma m). Se aplanan cerca de base y balizas.
DUNES = [
    (6.5, 3.6, 0.30, 1.0),
    (7.0, -2.2, 0.22, 0.8),
    (2.6, -2.8, 0.25, 0.9),
    (3.0, 1.9, 0.15, 0.7),
    (10.8, -1.4, 0.15, 0.6),
    (10.6, 1.6, 0.12, 0.5),
]
REFERENCE_ROUTE = 25.2                          # base -> B1 -> B2 -> base (m)

# --- Rover y tiempo.
GRAVITY = 3.71                                  # m/s^2 (Marte)
WHEEL_R, WHEEL_BASE = 0.1, 0.42                 # radio de rueda y separacion (m)
ROBOT_RADIUS = 0.3                              # para la busqueda de rutas (BFS)
SENSOR_HEIGHT = 0.3                             # altura de los rayos sobre el chasis
DT, TIMESTEP = 0.5, 0.02                        # 2 Hz de decision, 50 Hz de fisica
SUBSTEPS = 5                                    # revisiones de choque/arena por paso
V_MAX, W_MAX = 0.6, 1.2                         # m/s, rad/s
N_RAYS, RAY_RANGE = 9, 5.0
RAY_ANGLES = np.linspace(-math.pi / 2, math.pi / 2, N_RAYS)
TERRAIN_N = 121                                 # rejilla del heightfield (0.1 m)

# Acciones discretas: (velocidad pedida m/s, giro pedido rad/s).
ACTIONS = [
    ("avanzar", V_MAX, 0.0),
    ("avanzar + izq", 0.4, 0.8),
    ("avanzar + der", 0.4, -0.8),
    ("girar izq", 0.0, W_MAX),
    ("girar der", 0.0, -W_MAX),
]


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _in_rect(x, y, rect, margin=0.0):
    cx, cy, hx, hy = rect[:4]
    return abs(x - cx) <= hx + margin and abs(y - cy) <= hy + margin


def _circle_hits_rect(x, y, r, rect):
    cx, cy, hx, hy = rect[:4]
    dx = max(abs(x - cx) - hx, 0.0)
    dy = max(abs(y - cy) - hy, 0.0)
    return dx * dx + dy * dy < r * r


def terrain_heights(n=TERRAIN_N, dunes=None):
    """Altura del suelo (m) en una rejilla n x n; fila = y, columna = x."""
    xs, ys = np.linspace(0, MAP_W, n), np.linspace(-MAP_H / 2, MAP_H / 2, n)
    X, Y = np.meshgrid(xs, ys)
    H = np.zeros_like(X)
    for cx, cy, amp, sigma in DUNES if dunes is None else dunes:
        H += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    for px, py, r in [(BASE[0], BASE[1], 1.5), (*BEACONS[0], 0.9), (*BEACONS[1], 0.9)]:
        H *= np.clip((np.hypot(X - px, Y - py) - r) / 0.8, 0, 1)
    for _, cx, cy, hx, hy, _ in ROCKS:                 # rocas sobre suelo plano
        H *= np.clip((np.maximum(abs(X - cx) - hx, abs(Y - cy) - hy)) / 0.5, 0, 1)
    cx, cy, hx, hy = CRACK
    H[(abs(X - cx) <= hx) & (abs(Y - cy) <= hy)] = -CRACK_DEPTH
    return H


def _build_xml(rocks=None, dunes=None):
    rocks = ROCKS if rocks is None else rocks
    lo = -CRACK_DEPTH
    heights = terrain_heights(dunes=dunes)
    elevation = float(heights.max() - lo)
    walls = []
    for name, x, y, sx, sy in [("wall_n", MAP_W / 2, MAP_H / 2 + .075, MAP_W / 2 + .15, .075),
                               ("wall_s", MAP_W / 2, -MAP_H / 2 - .075, MAP_W / 2 + .15, .075),
                               ("wall_w", -.075, 0, .075, MAP_H / 2),
                               ("wall_e", MAP_W + .075, 0, .075, MAP_H / 2)]:
        walls.append(f'<geom name="{name}" type="box" pos="{x} {y} .2" size="{sx} {sy} .3" material="wall" group="0"/>')
    rocks_xml = [f'<geom name="rock_{i}" type="box" pos="{cx} {cy} {h / 2 - .05}" size="{hx} {hy} {h / 2 + .05}" '
                 f'material="rock" group="0"/>' for i, (_, cx, cy, hx, hy, h) in enumerate(rocks)]
    _, bx, by, bhx, bhy, bh = BLOCK
    xml = f"""
<mujoco model="robot_expedition_3d">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -{GRAVITY}" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>
    <hfield name="terrain" nrow="{TERRAIN_N}" ncol="{TERRAIN_N}" size="{MAP_W / 2} {MAP_H / 2} {elevation} 0.3"/>
    <texture name="mars" type="2d" builtin="checker" rgb1=".78 .48 .28" rgb2=".72 .43 .25" width="512" height="512"/>
    <material name="mars" texture="mars" texrepeat="12 12"/>
    <material name="wall" rgba=".48 .48 .52 1"/>
    <material name="rock" rgba=".55 .55 .58 1"/>
    <material name="rover" rgba=".9 .88 .82 1"/>
    <material name="wheel" rgba=".15 .17 .2 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="2 -4 12" dir="-.2 .3 -1" diffuse="1 .85 .7" castshadow="true"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{MAP_W / 2} 0 {lo}" material="mars" group="1"/>
    {''.join(walls)}
    {''.join(rocks_xml)}
    <geom name="block" type="box" pos="{bx} {by} -5" size="{bhx} {bhy} {bh / 2}" rgba=".94 .36 .3 1" group="0"/>
    <body name="rover" pos="{START[0]} {START[1]} .02">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size=".25 .16 .05" pos="0 0 .11" mass="8" material="rover" group="2"/>
      <geom name="mast" type="cylinder" size=".02 .1" pos=".14 0 .26" mass=".2" contype="0" conaffinity="0" material="wheel" group="2"/>
      <site name="sensor" pos=".16 0 {SENSOR_HEIGHT}" size=".03"/>
      <geom name="caster_f" type="sphere" size=".05" pos=".2 0 .08" mass=".2" condim="1" priority="1" friction="0 0 0" material="wheel" group="2"/>
      <geom name="caster_b" type="sphere" size=".05" pos="-.2 0 .08" mass=".2" condim="1" priority="1" friction="0 0 0" material="wheel" group="2"/>
      <body name="wheel_l" pos="0 {WHEEL_BASE / 2} {WHEEL_R}">
        <joint name="wheel_l" type="hinge" axis="0 1 0" damping=".02"/>
        <geom type="sphere" size="{WHEEL_R}" mass="1" friction="1.2 .02 .002" material="wheel" group="2"/>
      </body>
      <body name="wheel_r" pos="0 -{WHEEL_BASE / 2} {WHEEL_R}">
        <joint name="wheel_r" type="hinge" axis="0 1 0" damping=".02"/>
        <geom type="sphere" size="{WHEEL_R}" mass="1" friction="1.2 .02 .002" material="wheel" group="2"/>
      </body>
    </body>
    <camera name="overview" pos="6 -11 9" xyaxes="1 0 0 0 .6 .8"/>
  </worldbody>
  <actuator>
    <velocity name="wheel_l" joint="wheel_l" kv="3" ctrlrange="-8 8" forcerange="-1.5 1.5"/>
    <velocity name="wheel_r" joint="wheel_r" kv="3" ctrlrange="-8 8" forcerange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""
    return xml, heights, lo, elevation


def load_level_config(level_config=None):
    """Load a level mapping, level id, or YAML path using the shared contract."""
    defaults = {"beacons": 2, "active_rocks": 5, "dunes": True,
                "blocking": False, "ray_noise": 0.02, "dead_ray": None}
    if level_config is None:
        return defaults
    if isinstance(level_config, str):
        path = Path(level_config)
        if not path.exists():
            path = Path(__file__).parents[2] / "configs" / "levels" / f"{level_config}.yaml"
        with path.open(encoding="utf-8") as fh:
            level_config = yaml.safe_load(fh) or {}
    result = defaults | dict(level_config)
    if result["beacons"] not in (1, 2):
        raise ValueError("beacons debe ser 1 o 2")
    if not 0 <= result["active_rocks"] <= len(ROCKS):
        raise ValueError("active_rocks fuera del rango de rocas disponibles")
    return result


def load_scenario_set(scenario_set=None):
    """Carga y valida un conjunto versionado de escenarios por episodio."""
    if scenario_set is None:
        return None
    if isinstance(scenario_set, str):
        path = Path(scenario_set)
        if not path.exists():
            path = SCENARIOS_DIR / f"{scenario_set}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no existe el conjunto de escenarios {scenario_set}")
        with path.open(encoding="utf-8") as fh:
            scenario_set = yaml.safe_load(fh) or {}
    if not isinstance(scenario_set, dict):
        raise TypeError("scenario_set debe ser un id, una ruta YAML o un mapa")

    result = dict(scenario_set)
    scenarios = result.get("scenarios")
    if not result.get("scenario_set_id") or not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario_set requiere scenario_set_id y una lista no vacía de scenarios")

    normalized, known = [], set()
    for raw in scenarios:
        if not isinstance(raw, dict):
            raise TypeError("cada escenario debe ser un mapa")
        scenario = dict(raw)
        scenario_id = str(scenario.get("scenario_id", "")).strip()
        if not scenario_id or scenario_id in known:
            raise ValueError(f"scenario_id vacío o repetido: {scenario_id!r}")
        known.add(scenario_id)
        difficulty = scenario.get("difficulty")
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"{scenario_id}: difficulty debe ser easy, medium o hard")
        beacons = scenario.get("beacons")
        if not isinstance(beacons, list) or len(beacons) < 2:
            raise ValueError(f"{scenario_id}: se requieren al menos dos balizas")
        points = []
        for point in beacons:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{scenario_id}: cada baliza debe ser [x, y]")
            x, y = map(float, point)
            if not (VISIT_RADIUS <= x <= MAP_W - VISIT_RADIUS
                    and -MAP_H / 2 + VISIT_RADIUS <= y <= MAP_H / 2 - VISIT_RADIUS):
                raise ValueError(f"{scenario_id}: baliza {(x, y)} fuera del área segura")
            if _circle_hits_rect(x, y, VISIT_RADIUS, BASE) or _circle_hits_rect(x, y, VISIT_RADIUS, CRACK):
                raise ValueError(f"{scenario_id}: baliza {(x, y)} invade la base o la grieta")
            points.append((x, y))
        active_rocks = int(scenario.get("active_rocks", len(ROCKS)))
        rock_jitter = float(scenario.get("rock_jitter", 0.0))
        if not 0 <= active_rocks <= len(ROCKS):
            raise ValueError(f"{scenario_id}: active_rocks fuera de rango")
        if not 0 <= rock_jitter <= 2.0:
            raise ValueError(f"{scenario_id}: rock_jitter debe estar entre 0 y 2 m")
        normalized.append({
            **scenario,
            "scenario_id": scenario_id,
            "difficulty": difficulty,
            "beacons": points,
            "active_rocks": active_rocks,
            "rock_jitter": rock_jitter,
            "blocking": bool(scenario.get("blocking", False)),
        })
    result["scenarios"] = normalized
    return result


def environment_hash(config=None, environment=None):
    """SHA-256 reproducible del código del ambiente y configuración resuelta."""
    payload = inspect.getsource(inspect.getmodule(BeaconExpeditionEnv))
    options = dict(environment or {})
    if options.get("scenario_set") is not None:
        options["scenario_set"] = load_scenario_set(options["scenario_set"])
    resolved = {"level": load_level_config(config), "environment": options}
    payload += json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BeaconExpeditionEnv(gym.Env):
    """Visitar las balizas configuradas y volver a la base (MuJoCo 3D)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, max_steps=400, ray_noise=None, rock_jitter=0.0,
                 perturbation=False, render_mode=None, level_config=None,
                 level=None, config=None, scenario_set=None):
        super().__init__()
        self.level_config = load_level_config(level_config or config or level)
        self.scenario_set = load_scenario_set(scenario_set)
        self.beacon_count = int(self.level_config["beacons"])
        self.active_rocks = int(self.level_config["active_rocks"])
        self.dunes_enabled = bool(self.level_config["dunes"])
        self.blocking = bool(self.level_config["blocking"])
        self.ray_noise = float(self.level_config["ray_noise"] if ray_noise is None else ray_noise)
        self.dead_ray = self.level_config.get("dead_ray")
        self.max_steps, self.rock_jitter = max_steps, rock_jitter
        self.perturbation, self.render_mode = perturbation or self.blocking, render_mode
        self._level_rocks = ROCKS[:self.active_rocks]
        xml, self.heights, self._terrain_lo, elevation = _build_xml(
            rocks=self._level_rocks, dunes=DUNES if self.dunes_enabled else [])
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.hfield_data[:] = ((self.heights - self._terrain_lo) / elevation).ravel()
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        gid = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self._rover = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        self._sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "sensor")
        self._rock_ids = [gid(f"rock_{i}") for i in range(len(self._level_rocks))]
        self._block_id = gid("block")
        self._rock_pos0 = self.model.geom_pos[self._rock_ids].copy()
        self._obstacle_ids = set(self._rock_ids) | {self._block_id} | {gid(w) for w in ("wall_n", "wall_s", "wall_w", "wall_e")}
        self._is_obstacle = np.isin(np.arange(self.model.ngeom), list(self._obstacle_ids))
        self._is_rover = self.model.geom_bodyid != 0
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._substeps = int(round(DT / TIMESTEP))
        self.action_space = spaces.Discrete(len(ACTIONS))
        low = np.array([0] * N_RAYS + [0, -1, -1, -1, -1, 0, 0, 0, 0, -math.pi, -math.pi], dtype=np.float32)
        high = np.array([RAY_RANGE] * N_RAYS + [17, 1, 1, 1, 1, 1, 1, 1, 1, math.pi, math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self._rock_names = [r[0] for r in self._level_rocks]
        self.beacons = list(BEACONS[:self.beacon_count])
        self.rocks = [r[1:] for r in self._level_rocks]
        self.scenario_id = "fixed"
        self.scenario_difficulty = "fixed"
        self.episode_blocking = self.perturbation
        self.episode_rock_jitter = self.rock_jitter
        self.block_active = False

    # ------------------------------------------------------------- consultas
    @property
    def x(self): return float(self.data.qpos[0])
    @property
    def y(self): return float(self.data.qpos[1])
    @property
    def z(self): return float(self.data.qpos[2])
    @property
    def quat(self): return self.data.qpos[3:7]

    @property
    def theta(self):
        w, qx, qy, qz = self.quat
        return math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    @property
    def tilt(self):
        """(cabeceo, alabeo) en rad."""
        w, qx, qy, qz = self.quat
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx))))
        roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        return pitch, roll

    @property
    def upright(self):
        return float(self.data.xmat[self._rover][8])

    def ground(self, x, y):
        """Altura del terreno en (x, y) (vecino mas cercano de la rejilla)."""
        i = int(round((y + MAP_H / 2) / MAP_H * (TERRAIN_N - 1)))
        j = int(round(x / MAP_W * (TERRAIN_N - 1)))
        return float(self.heights[min(max(i, 0), TERRAIN_N - 1), min(max(j, 0), TERRAIN_N - 1)])

    def obstacles(self):
        rocks = list(self.rocks)
        if self.block_active:
            rocks.append(BLOCK[1:])
        return rocks

    def target(self):
        return self.beacons[self.stage - 1] if self.stage <= self.beacon_count else BASE[:2]

    def _distance_to_target(self):
        tx, ty = self.target()
        return math.hypot(tx - self.x, ty - self.y)

    def rays(self):
        """9 rayos horizontales en 3D desde el sensor del rover (mj_ray)."""
        origin = self.data.site_xpos[self._sensor].copy()
        out = np.empty(N_RAYS, dtype=np.float32)
        geom = np.zeros(1, dtype=np.int32)
        for i, rel in enumerate(RAY_ANGLES):
            a = self.theta + rel
            vec = np.array([math.cos(a), math.sin(a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, origin, vec, self._ray_groups, 1, self._rover, geom)
            out[i] = RAY_RANGE if d < 0 else min(d, RAY_RANGE)
        if self.ray_noise > 0:
            out += self.np_random.normal(0, self.ray_noise, N_RAYS).astype(np.float32)
        if self.dead_ray is not None:
            out[int(self.dead_ray)] = RAY_RANGE
        return np.clip(out, 0, RAY_RANGE)

    # ------------------------------------------------------ rutas y rocas
    def _route_exists(self, cell=0.25):
        """BFS en 2D: base -> B1 -> B2 -> base sigue siendo alcanzable."""
        nx, ny = int(MAP_W / cell), int(MAP_H / cell)
        r = ROBOT_RADIUS
        def free(x, y):
            if x - r < 0 or x + r > MAP_W or y - r < -MAP_H / 2 or y + r > MAP_H / 2:
                return False
            return not any(_circle_hits_rect(x, y, r, rect) for rect in self.obstacles())
        grid = np.array([[free((i + .5) * cell, (j + .5) * cell - MAP_H / 2) for j in range(ny)] for i in range(nx)])
        def idx(p): return int(p[0] / cell), int((p[1] + MAP_H / 2) / cell)
        start = idx(BASE[:2]); seen = {start}; queue = [start]
        while queue:
            i, j = queue.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < nx and 0 <= b < ny and grid[a, b] and (a, b) not in seen:
                    seen.add((a, b)); queue.append((a, b))
        return all(idx(b) in seen for b in self.beacons)

    def _place_rocks(self, rocks):
        self.rocks = list(rocks)
        for gid in self._rock_ids:
            self.model.geom_pos[gid][2] = -5.0
        for index, (gid, (cx, cy, *_)) in enumerate(zip(self._rock_ids, self.rocks)):
            self.model.geom_pos[gid] = [cx, cy, self._rock_pos0[index, 2]]

    def _jitter_rocks(self, base_rocks, jitter):
        """Mueve rocas sin tapar base, balizas, grieta ni la ruta completa."""
        keep_clear = [BASE[:2], *self.beacons, CRACK[:2]]
        for _ in range(50):
            rocks = [(cx + self.np_random.uniform(-1, 1) * jitter,
                      cy + self.np_random.uniform(-1, 1) * jitter, hx, hy, h)
                     for cx, cy, hx, hy, h in base_rocks]
            if all(not _circle_hits_rect(px, py, 1.0, r) for r in rocks for px, py in keep_clear):
                self._place_rocks(rocks)
                if self._route_exists():
                    return
        self._place_rocks(base_rocks)

    def _select_scenario(self, seed=None, options=None):
        requested = (options or {}).get("scenario_id")
        if self.scenario_set is None:
            if requested is not None:
                raise ValueError("scenario_id requiere configurar scenario_set")
            self.scenario_id = "fixed"
            self.scenario_difficulty = "fixed"
            self.beacons = list(BEACONS[:self.beacon_count])
            self.episode_blocking = self.perturbation
            self.episode_rock_jitter = self.rock_jitter
            return list(self._level_rocks)

        scenarios = self.scenario_set["scenarios"]
        if requested is None:
            index = int(seed) % len(scenarios) if seed is not None else int(self.np_random.integers(len(scenarios)))
            scenario = scenarios[index]
        else:
            scenario = next((item for item in scenarios if item["scenario_id"] == requested), None)
            if scenario is None:
                raise ValueError(f"no existe el escenario {requested!r} en {self.scenario_set['scenario_set_id']}")
        self.scenario_id = scenario["scenario_id"]
        self.scenario_difficulty = scenario["difficulty"]
        self.beacons = list(scenario["beacons"][:self.beacon_count])
        self.episode_blocking = self.perturbation or bool(scenario["blocking"])
        self.episode_rock_jitter = max(self.rock_jitter, float(scenario["rock_jitter"]))
        active = min(self.active_rocks, int(scenario["active_rocks"]))
        return list(self._level_rocks[:active])

    def _set_block(self, active):
        self.block_active = active
        self.model.geom_pos[self._block_id][2] = BLOCK[5] / 2 if active else -5.0

    def set_pose(self, x, y, theta=0.0, speed=0.0):
        """Coloca el rover sobre el terreno (para pruebas y escenarios)."""
        self.data.qpos[:3] = [x, y, self.ground(x, y) + 0.005]
        self.data.qpos[3:7] = [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]
        self.data.qvel[:] = 0
        self.data.qvel[0:2] = [speed * math.cos(theta), speed * math.sin(theta)]
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_block(False)
        episode_rocks = [r[1:] for r in self._select_scenario(seed, options)]
        self._place_rocks(episode_rocks)
        if self.episode_rock_jitter > 0:
            self._jitter_rocks(episode_rocks, self.episode_rock_jitter)
        self.set_pose(*START)
        for _ in range(30):                              # el rover se asienta
            mujoco.mj_step(self.model, self.data)
        self.battery, self.stage, self.visited = 100.0, 1, [False, False]
        self.steps = self.collisions = 0
        self.path_length = 0.0
        return self._observation(), self._info(None)

    def _forward_speed(self):
        heading = self.data.xmat[self._rover].reshape(3, 3)[:, 0]
        return float(np.dot(self.data.qvel[0:3], heading))

    def step(self, action):
        _, v_cmd, w_cmd = ACTIONS[int(action)]
        d_before = self._distance_to_target()
        # Ruedas esfericas: contacto estable con el heightfield. Cinematica diferencial: velocidad de cada rueda (rad/s).
        self.data.ctrl[:] = [(v_cmd - w_cmd * WHEEL_BASE / 2) / WHEEL_R,
                             (v_cmd + w_cmd * WHEEL_BASE / 2) / WHEEL_R]
        collision = False
        # 5 bloques de 0.1 s: la fisica avanza en C y Python revisa entre bloques.
        for _ in range(SUBSTEPS):
            x0, y0 = self.x, self.y
            # Arena suelta: fuerza de arrastre horizontal.
            drag = -12.0 * self.data.qvel[0:2] if _in_rect(x0, y0, SAND) else (0.0, 0.0)
            self.data.xfrc_applied[self._rover, 0:2] = drag
            mujoco.mj_step(self.model, self.data, nstep=self._substeps // SUBSTEPS)
            self.path_length += math.hypot(self.x - x0, self.y - y0)
            if self.data.ncon:
                g = self.data.contact.geom
                obst, rover = self._is_obstacle, self._is_rover
                collision |= bool(np.any((obst[g[:, 0]] & rover[g[:, 1]]) | (obst[g[:, 1]] & rover[g[:, 0]])))
        v, w = self._forward_speed(), float(self.data.qvel[5])

        energy = min(1.0, (v / V_MAX) ** 2 + 0.25 * (w / W_MAX) ** 2)
        self.battery = max(0.0, self.battery - (0.1 + 0.4 * energy))
        self.steps += 1
        self.collisions += int(collision)

        progress = d_before - self._distance_to_target()
        reward = 1.0 * progress - 0.01 - 0.02 * energy - 0.5 * collision
        event = None
        if self.stage <= self.beacon_count:
            bx, by = self.beacons[self.stage - 1]
            if math.hypot(bx - self.x, by - self.y) < VISIT_RADIUS:
                self.visited[self.stage - 1] = True
                event = f"visita B{self.stage}"
                reward += 10.0
                # La etapa 3 siempre significa "volver a la base", en todos los niveles:
                # asi la observacion (one-hot) y los cuadros tienen el mismo significado.
                self.stage = self.stage + 1 if self.stage < self.beacon_count else 3
                if self.stage == 2 and self.episode_blocking:
                    self._set_block(True)

        success = self.stage == 3 and _in_rect(self.x, self.y, BASE) and self.battery > 0
        end = None
        if success:
            reward += 50.0 + 20.0 * self.battery / 100.0
            end = "exito"
        elif _in_rect(self.x, self.y, CRACK) or self.z < -0.15:
            reward -= 50.0; end = "grieta"
        elif self.upright < 0.3:
            reward -= 50.0; end = "volco"
        elif self.battery <= 0:
            reward -= 50.0; end = "bateria"
        terminated = end is not None
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            end = "tiempo"
        info = self._info(end)
        info.update(collision=collision, event=event, energy=energy, progress=progress)
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self):
        tx, ty = self.target()
        rel = _wrap(math.atan2(ty - self.y, tx - self.x) - self.theta)
        stage = [0.0, 0.0, 0.0]; stage[self.stage - 1] = 1.0
        self.last_rays = self.rays()
        v = max(-1.0, min(1.0, self._forward_speed() / V_MAX))
        w = max(-1.0, min(1.0, float(self.data.qvel[5]) / W_MAX))
        tail = [self._distance_to_target(), math.sin(rel), math.cos(rel), v, w,
                self.battery / 100.0, *stage, *self.tilt]
        return np.concatenate([self.last_rays, np.array(tail, dtype=np.float32)]).astype(np.float32)

    def _info(self, end):
        return {"x": self.x, "y": self.y, "z": self.z, "theta": self.theta, "stage": self.stage,
                "visited": list(self.visited), "battery": self.battery,
                "collisions": self.collisions, "path_length": self.path_length,
                "success": end == "exito", "end": end,
                "scenario_id": self.scenario_id, "difficulty": self.scenario_difficulty,
                "mission_progress": (sum(self.visited) + (end == "exito")) / (self.beacon_count + 1)}

    def reference_route(self):
        """Distancia recta base -> balizas del episodio -> base."""
        points = [BASE[:2], *self.beacons, BASE[:2]]
        return float(sum(math.dist(a, b) for a, b in zip(points, points[1:])))

    def map_spec(self):
        """Descripcion del mundo 3D para la visualizacion con Three.js."""
        return {
            "size": [MAP_W, MAP_H], "base": BASE, "beacons": self.beacons,
            "visit_radius": VISIT_RADIUS, "crack": CRACK, "crack_depth": CRACK_DEPTH, "sand": SAND,
            "rocks": [[name, *rect] for name, rect in zip(self._rock_names, self.rocks)],
            "block": list(BLOCK), "wheel_radius": WHEEL_R, "wheel_base": WHEEL_BASE,
            "sensor_height": SENSOR_HEIGHT, "gravity": GRAVITY,
            "scenario_id": self.scenario_id, "difficulty": self.scenario_difficulty,
            "terrain": {"n": TERRAIN_N, "heights": np.round(self.heights, 3).ravel().tolist()},
            "ray_angles": RAY_ANGLES.round(4).tolist(), "ray_range": RAY_RANGE,
            "actions": [a[0] for a in ACTIONS], "dt": DT,
        }

    def render(self):
        """Imagen 3D de MuJoCo (rgb_array) desde la camara general."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
