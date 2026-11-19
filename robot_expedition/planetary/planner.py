"""Guía planificadora para el rover planetario.

Conoce solo el *mapa de misión* estático (área activa, base, rocas, grieta,
cráteres, dunas y zonas de baja tracción) y su pose de aterrizaje. Nunca recibe
la posición de las muestras: las descubre con la cámara de la observación
pública y estima dónde están a partir del rumbo y la distancia observados.

Se ubica por odometría (velocidad y giro de la observación más reciente). Sin
choques, el error medido es de unos 10 cm tras 17 m, por eso la ruta se traza
con A* manteniendo márgenes de seguridad alrededor de los obstáculos.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from robot_expedition.envs.beacon_expedition import BASE, CRACK, DT, RAY_RANGE, V_MAX, W_MAX, _wrap

OBS_SIZE = 33
COLLECT, FORWARD, LEFT_ARC, RIGHT_ARC, TURN_LEFT, TURN_RIGHT, REVERSE = 5, 0, 1, 2, 3, 4, 6
CELL = 0.2
ROVER_CLEARANCE = 0.42          # radio del rover + margen para no rozar
WALL_CLEARANCE = 0.35
HAZARD_CLEARANCE = 0.55         # grieta y cráteres: caer termina la misión
COLLECT_DISTANCE = 0.6          # el entorno recolecta hasta 0.68 m


def latest(observation) -> np.ndarray:
    """La observación de la API apila el historial; la lectura actual es la última."""
    return np.asarray(observation, dtype=np.float32)[-OBS_SIZE:]


@dataclass
class MissionMap:
    x_max: float
    y_half: float
    rocks: list[tuple[float, float, float, float]]
    hazards_rect: list[tuple[float, float, float, float]]
    hazards_circle: list[tuple[float, float, float]]
    slow_circle: list[tuple[float, float, float]] = field(default_factory=list)
    slow_rect: list[tuple[float, float, float, float]] = field(default_factory=list)

    @classmethod
    def from_env(cls, env) -> "MissionMap":
        """Extrae del escenario solo la información estática de la misión (sin muestras)."""
        scenario = env.scenario
        rocks = [tuple(float(v) for v in rock[:4]) for rock in env.rocks]
        # Cráter = campana gaussiana (cx, cy, amplitud<0, sigma). La misión termina si el rover
        # baja de -0.15 m: se bloquea ese núcleo letal y los bordes transitables solo cuestan más.
        craters = []
        for cx, cy, amplitude, sigma in (tuple(float(v) for v in c[:4]) for c in scenario.craters):
            depth = abs(amplitude)
            radius = sigma * math.sqrt(2.0 * math.log(depth / 0.15)) if depth > 0.15 else 0.0
            craters.append((cx, cy, radius))
        dunes = [(float(d[0]), float(d[1]), float(d[3])) for d in scenario.dunes]
        zones = [tuple(float(v) for v in z[:4]) for z in scenario.traction_zones]
        return cls(float(scenario.active_x_max), float(scenario.active_y_half), rocks,
                   [tuple(float(v) for v in CRACK)], craters, dunes, zones)

    def __post_init__(self) -> None:
        self.nx = int(math.ceil(self.x_max / CELL))
        self.ny = int(math.ceil(2 * self.y_half / CELL))
        xs = (np.arange(self.nx) + 0.5) * CELL
        ys = (np.arange(self.ny) + 0.5) * CELL - self.y_half
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.X, self.Y = X, Y
        blocked = (X < WALL_CLEARANCE) | (X > self.x_max - WALL_CLEARANCE)
        blocked |= (np.abs(Y) > self.y_half - WALL_CLEARANCE)
        for cx, cy, hx, hy in self.rocks:
            blocked |= _rect_distance(X, Y, cx, cy, hx, hy) < ROVER_CLEARANCE
        for cx, cy, hx, hy in self.hazards_rect:
            blocked |= _rect_distance(X, Y, cx, cy, hx, hy) < HAZARD_CLEARANCE
        cost = np.ones_like(X)
        risky = np.zeros_like(X, dtype=bool)
        for cx, cy, r in self.hazards_circle:
            distance = np.hypot(X - cx, Y - cy)
            blocked |= distance < r + 0.05
            cost += 6.0 * (distance < r + 0.5) + 3.0 * (distance < r + 0.9)
            risky |= distance < r + 1.0
        for cx, cy, hx, hy in self.hazards_rect:
            risky |= _rect_distance(X, Y, cx, cy, hx, hy) < HAZARD_CLEARANCE + 0.5
        self.risky = risky
        for cx, cy, r in self.slow_circle:
            cost += 3.0 * (np.hypot(X - cx, Y - cy) < r + 0.2)
        for cx, cy, hx, hy in self.slow_rect:
            cost += 2.0 * (_rect_distance(X, Y, cx, cy, hx, hy) < 0.2)
        # Preferir pasar lejos de obstáculos: el roce degrada la odometría.
        near = np.zeros_like(X, dtype=bool)
        for cx, cy, hx, hy in self.rocks:
            near |= _rect_distance(X, Y, cx, cy, hx, hy) < ROVER_CLEARANCE + 0.3
        cost += 1.5 * near
        self.blocked, self.cost = blocked, cost

    def cell(self, x: float, y: float) -> tuple[int, int]:
        i = min(self.nx - 1, max(0, int(x / CELL)))
        j = min(self.ny - 1, max(0, int((y + self.y_half) / CELL)))
        return i, j

    def center(self, i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * CELL, (j + 0.5) * CELL - self.y_half

    def nearest_free(self, x: float, y: float) -> tuple[int, int]:
        i, j = self.cell(x, y)
        if not self.blocked[i, j]:
            return i, j
        free = np.argwhere(~self.blocked)
        k = int(np.argmin((free[:, 0] - i) ** 2 + (free[:, 1] - j) ** 2))
        return int(free[k, 0]), int(free[k, 1])

    def free(self, x: float, y: float) -> bool:
        i, j = self.cell(x, y)
        return not self.blocked[i, j]

    def path(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        s, g = self.nearest_free(*start), self.nearest_free(*goal)
        frontier = [(0.0, s)]
        came: dict = {s: None}
        best = {s: 0.0}
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == g:
                break
            ci, cj = current
            for di, dj, length in steps:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny) or self.blocked[ni, nj]:
                    continue
                if di and dj and (self.blocked[ci + di, cj] or self.blocked[ci, cj + dj]):
                    continue
                score = best[current] + length * float(self.cost[ni, nj])
                if score < best.get((ni, nj), math.inf):
                    best[(ni, nj)] = score
                    came[(ni, nj)] = current
                    heapq.heappush(frontier, (score + math.hypot(ni - g[0], nj - g[1]), (ni, nj)))
        if g not in came:
            return None
        cells, node = [], g
        while node is not None:
            cells.append(node)
            node = came[node]
        return [self.center(*c) for c in reversed(cells)]

    def path_length(self, start, goal) -> float:
        route = self.path(start, goal)
        if route is None:
            return math.inf
        return sum(math.dist(a, b) for a, b in zip(route, route[1:]))

    def viewpoints(self, spacing: float) -> list[tuple[float, float]]:
        """Puntos de observación que cubren el área con la cámara (alcance ~ visibilidad)."""
        points = []
        safe = np.argwhere(~self.blocked & ~self.risky)
        for x in np.arange(spacing * 0.75, self.x_max, spacing):
            for y in np.arange(-self.y_half + spacing * 0.6, self.y_half, spacing):
                i, j = self.cell(float(x), float(y))
                # El punto de observación va a la celda segura más cercana (nunca al borde de un cráter).
                k = int(np.argmin((safe[:, 0] - i) ** 2 + (safe[:, 1] - j) ** 2))
                px, py = self.center(int(safe[k, 0]), int(safe[k, 1]))
                if math.dist((px, py), (x, y)) < spacing / 2:
                    points.append((px, py))
        return points


SENSOR_OFFSET = 0.16
RAY_ANGLES = np.linspace(-math.pi / 2, math.pi / 2, 9)
WALLS = ((-0.5, 0.0, 0.5, 7.0), (12.5, 0.0, 0.5, 7.0), (6.0, -6.5, 7.0, 0.5), (6.0, 6.5, 7.0, 0.5))


def expected_rays(poses: np.ndarray, rects: np.ndarray, max_range: float) -> np.ndarray:
    """Rayos LIDAR esperados (K, 9) desde poses (K, 3) contra rectángulos alineados (R, 4)."""
    ox = poses[:, 0] + SENSOR_OFFSET * np.cos(poses[:, 2])
    oy = poses[:, 1] + SENSOR_OFFSET * np.sin(poses[:, 2])
    angles = poses[:, 2:3] + RAY_ANGLES[None, :]
    dx, dy = np.cos(angles)[..., None], np.sin(angles)[..., None]          # (K, 9, 1)
    ox, oy = ox[:, None, None], oy[:, None, None]
    cx, cy, hx, hy = (rects[:, k][None, None, :] for k in range(4))       # (1, 1, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx1, tx2 = (cx - hx - ox) / dx, (cx + hx - ox) / dx
        ty1, ty2 = (cy - hy - oy) / dy, (cy + hy - oy) / dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
    distance = np.where(hit, np.maximum(tmin, 0.0), np.inf).min(axis=2)
    return np.minimum(distance, max_range)


def _rect_distance(X, Y, cx, cy, hx, hy):
    dx = np.maximum(np.abs(X - cx) - hx, 0.0)
    dy = np.maximum(np.abs(Y - cy) - hy, 0.0)
    return np.hypot(dx, dy)


class PlanetaryMissionPlanner:
    """Planificador A* + cobertura con cámara; actúa solo con observación pública."""

    name = "planificador de misión"

    def __init__(self):
        self.map: MissionMap | None = None
        self.pose = [BASE[0], BASE[1], 0.0]
        self.started = False
        self.caution = False
        self.targets: list[list[float]] = []     # [x, y, avistamientos]
        self.pending: list[tuple[float, float]] = []
        self.view: tuple[float, float] | None = None
        self.route: list[tuple[float, float]] = []
        self.goal: tuple[float, float] | None = None
        self.replan_in = 0
        self.stuck = 0
        self.escape: list[int] = []
        self.collected = 0.0            # fracción recolectada observada
        self.collect_attempts = 0
        self.steps = 0

    # --------------------------------------------------------- integración
    def bind(self, env) -> "PlanetaryMissionPlanner":
        """Recibe el mapa de misión y la pose de aterrizaje del episodio."""
        self.map = MissionMap.from_env(env)
        self._rects = np.array(list(self.map.rocks) + list(WALLS), dtype=np.float64)
        self.pose = [float(env.x), float(env.y), float(env.theta)]
        visibility = float(getattr(env, "visibility_range", RAY_RANGE))
        spacing = max(2.5, min(4.5, visibility * 0.85))
        self.pending = self._order(self.map.viewpoints(spacing))
        return self

    def _order(self, points):
        ordered, here = [], (self.pose[0], self.pose[1])
        points = [p for p in points if math.dist(p, (BASE[0], BASE[1])) > 1.6]
        while points:
            nxt = min(points, key=lambda p: math.dist(here, p))
            ordered.append(nxt)
            points.remove(nxt)
            here = nxt
        return ordered

    # --------------------------------------------------------------- pasos
    def __call__(self, observation) -> int:
        o = latest(observation)
        self.steps += 1
        if self.started:
            self._odometry(o)
        self.started = True
        # Zona de riesgo (grieta o cráter cerca): quien use este guía debe obedecerlo aquí.
        self.caution = bool(self.map is not None and self.map.risky[self.map.cell(self.pose[0], self.pose[1])])
        if self.map is None:            # sin mapa (uso fuera de la API): solo reacción local
            if self.escape:
                return self.escape.pop(0)
            if self._is_stuck(o):
                return self._start_escape(o)
            return self._reactive(o)

        fraction = float(o[15])
        if fraction > self.collected + 1e-6:
            self.collected = fraction
            self._forget_nearest_target()
            self.collect_attempts = 0
            self.goal = None

        returning = float(o[15]) >= 0.999
        target_visible = not returning and (float(o[10]) != 0.0 or float(o[11]) != 0.0)
        if target_visible:
            self._remember_target(o)
            if float(o[22]) <= COLLECT_DISTANCE and self.collect_attempts < 3:
                self.collect_attempts += 1
                return COLLECT
        if self.escape:
            return self.escape.pop(0)
        if self._is_stuck(o):
            return self._start_escape(o)

        goal = self._choose_goal(returning)
        if goal is None:
            return self._reactive(o)
        if target_visible and float(o[22]) < 2.0 and not returning:
            # Aproximación final con el rumbo medido por la cámara (sin error de odometría).
            return self._steer(math.atan2(float(o[10]), float(o[11])), float(o[22]), o)
        if returning and math.dist(self.pose[:2], (BASE[0], BASE[1])) < 1.4:
            return self._steer(math.atan2(float(o[10]), float(o[11])), float(o[9]), o)
        if goal != self.goal or self.replan_in <= 0 or not self.route:
            self.goal = goal
            self.route = self.map.path((self.pose[0], self.pose[1]), goal) or []
            self.replan_in = 12
        self.replan_in -= 1
        return self._follow(o)

    # ------------------------------------------------------------ estimación
    def _odometry(self, o) -> None:
        v, w = float(o[12]) * V_MAX, float(o[13]) * W_MAX
        self.pose[2] = _wrap(self.pose[2] + w * DT)
        self.pose[0] += v * DT * math.cos(self.pose[2])
        self.pose[1] += v * DT * math.sin(self.pose[2])
        if self.map is not None:
            self._localize(o)

    _GRID = None

    def _localize(self, o) -> None:
        """Corrige la odometría comparando el LIDAR con rocas y paredes conocidas."""
        rays = np.asarray(o[:9], dtype=np.float64)
        visibility = float(o[28]) * RAY_RANGE if float(o[28]) > 0 else RAY_RANGE
        informative = rays < visibility - 0.05
        if int(informative.sum()) < 3:
            return                       # campo abierto: nada con qué comparar
        if PlanetaryMissionPlanner._GRID is None:
            d = np.linspace(-0.3, 0.3, 7)
            a = np.linspace(-0.12, 0.12, 7)
            PlanetaryMissionPlanner._GRID = np.array(np.meshgrid(d, d, a, indexing="ij")).reshape(3, -1).T
        poses = np.asarray(self.pose)[None, :] + PlanetaryMissionPlanner._GRID
        predicted = expected_rays(poses, self._rects, visibility)
        # Error robusto: los rayos ruidosos o bloqueados por algo no mapeado pesan poco.
        residual = np.minimum(np.abs(predicted - rays[None, :]), 0.6)
        score = residual.mean(axis=1) + 0.05 * np.linalg.norm(PlanetaryMissionPlanner._GRID[:, :2], axis=1)
        best = int(np.argmin(score))
        current = int(np.argmin(np.linalg.norm(PlanetaryMissionPlanner._GRID, axis=1)))
        if score[best] < score[current] - 0.01 and score[best] < 0.25:
            gain = 0.5
            self.pose[0] += gain * (poses[best, 0] - self.pose[0])
            self.pose[1] += gain * (poses[best, 1] - self.pose[1])
            self.pose[2] = _wrap(self.pose[2] + gain * _wrap(poses[best, 2] - self.pose[2]))

    def _remember_target(self, o) -> None:
        bearing = self.pose[2] + math.atan2(float(o[10]), float(o[11]))
        distance = float(o[22])
        x = self.pose[0] + distance * math.cos(bearing)
        y = self.pose[1] + distance * math.sin(bearing)
        for item in self.targets:
            if math.dist(item[:2], (x, y)) < 0.9:
                weight = item[2]
                item[0] = (item[0] * weight + x) / (weight + 1)
                item[1] = (item[1] * weight + y) / (weight + 1)
                item[2] = min(weight + 1, 6)
                return
        self.targets.append([x, y, 1])

    def _forget_nearest_target(self) -> None:
        if self.targets:
            nearest = min(self.targets, key=lambda t: math.dist(t[:2], self.pose[:2]))
            if math.dist(nearest[:2], self.pose[:2]) < 1.5:
                self.targets.remove(nearest)

    def _choose_goal(self, returning: bool):
        here = (self.pose[0], self.pose[1])
        if returning:
            return (BASE[0], BASE[1])
        if self.targets:
            target = min(self.targets, key=lambda t: math.dist(t[:2], here))
            if math.dist(target[:2], here) < 0.5:
                # Llegamos a la estimación y la cámara no la confirma: era un error.
                self.targets.remove(target)
            else:
                return target[0], target[1]
        # Puntos ya cubiertos por la cámara (pasamos cerca) dejan de ser metas.
        self.pending = [p for p in self.pending if math.dist(p, here) >= 0.8]
        if self.view is not None and self.view not in self.pending:
            self.view = None
        if self.view is None and self.pending:
            # Se elige por longitud de ruta (A*), no en línea recta: en mapas con
            # corredores la distancia euclídea engaña. La meta se mantiene hasta
            # alcanzarla, sin oscilar entre dos candidatos equidistantes.
            candidates = sorted(self.pending, key=lambda p: math.dist(p, here))[:5]
            lengths = [(self.map.path_length(here, p), p) for p in candidates]
            reachable = [item for item in lengths if math.isfinite(item[0])]
            if not reachable:
                self.pending = [p for p in self.pending if p not in candidates]
                return self._choose_goal(returning)
            self.view = min(reachable)[1]
        return self.view

    # --------------------------------------------------------------- control
    def _follow(self, o) -> int:
        here = (self.pose[0], self.pose[1])
        while len(self.route) > 1 and math.dist(self.route[0], here) < 0.35:
            self.route.pop(0)
        if not self.route:
            return self._reactive(o)
        look = self.route[min(len(self.route) - 1, 2)]
        heading = math.atan2(look[1] - here[1], look[0] - here[0])
        return self._steer(_wrap(heading - self.pose[2]), math.dist(look, here), o)

    def _steer(self, error: float, distance: float, o) -> int:
        front = float(np.min(o[3:6]))
        if abs(error) > 0.55:
            return TURN_LEFT if error > 0 else TURN_RIGHT
        if front < 0.4:
            return REVERSE
        if abs(error) > 0.18:
            return LEFT_ARC if error > 0 else RIGHT_ARC
        return FORWARD

    def _is_stuck(self, o) -> bool:
        if abs(float(o[12])) < 0.05 and abs(float(o[13])) < 0.05:
            self.stuck += 1
        else:
            self.stuck = 0
        return self.stuck >= 3

    def _start_escape(self, o) -> int:
        self.stuck = 0
        left, right = float(np.sum(o[:4])), float(np.sum(o[5:9]))
        turn = TURN_LEFT if left >= right else TURN_RIGHT
        self.escape = [REVERSE, REVERSE, turn, turn]
        self.route = []
        return self.escape.pop(0)

    @staticmethod
    def _reactive(o) -> int:
        front = float(np.min(o[3:6]))
        if front < 0.7:
            return TURN_LEFT if float(np.sum(o[:4])) >= float(np.sum(o[5:9])) else TURN_RIGHT
        return FORWARD
