"""Generación procedural reproducible y validación de escenarios marcianos."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from robot_expedition.envs.beacon_expedition import (
    BASE,
    CRACK,
    MAP_H,
    MAP_W,
    ROBOT_RADIUS,
    TERRAIN_N,
)

from .curriculum import CurriculumLevel, level_spec
from .profiles import get_planet


GENERATOR_VERSION = "planetary-scenarios/1.0.0"
SAMPLE_TYPES = ("basalto", "hematita", "sulfato")


def _in_rect(x: float, y: float, rect: tuple, margin: float = 0.0) -> bool:
    cx, cy, hx, hy = rect[:4]
    return abs(x - cx) <= hx + margin and abs(y - cy) <= hy + margin


def _circle_hits_rect(x: float, y: float, radius: float, rect: tuple) -> bool:
    cx, cy, hx, hy = rect[:4]
    dx = max(abs(x - cx) - hx, 0.0)
    dy = max(abs(y - cy) - hy, 0.0)
    return dx * dx + dy * dy < radius * radius


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    sample_type: str
    x: float
    y: float
    target: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanetaryScenario:
    seed: int
    scenario_id: str
    generator_version: str
    planet: dict
    difficulty_level: int
    difficulty_name: str
    active_x_max: float
    active_y_half: float
    target_type: str
    requested_samples: int
    capacity: int
    rocks: list[tuple[float, float, float, float, float]]
    samples: list[SampleSpec]
    dunes: list[tuple[float, float, float, float]]
    craters: list[tuple[float, float, float, float]]
    traction_zones: list[tuple[float, float, float, float, float]]
    traction: float
    wind_speed_m_s: float
    wind_direction_rad: float
    dust_opacity: float
    temperature_c: float
    battery_budget: float
    max_steps: int
    terrain_heights: np.ndarray
    generation_attempt: int = 0

    @property
    def target_samples(self) -> list[SampleSpec]:
        return [sample for sample in self.samples if sample.target]

    def to_dict(self, *, include_heightfield: bool = True) -> dict:
        data = {
            "schema_version": 1,
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "generator_version": self.generator_version,
            "planet": self.planet,
            "difficulty_level": self.difficulty_level,
            "difficulty_name": self.difficulty_name,
            "active_extent": [self.active_x_max, self.active_y_half * 2],
            "target_type": self.target_type,
            "requested_samples": self.requested_samples,
            "capacity": self.capacity,
            "rocks": [list(rock) for rock in self.rocks],
            "samples": [sample.to_dict() for sample in self.samples],
            "dunes": [list(item) for item in self.dunes],
            "craters": [list(item) for item in self.craters],
            "traction_zones": [list(item) for item in self.traction_zones],
            "traction": self.traction,
            "wind_speed_m_s": self.wind_speed_m_s,
            "wind_direction_rad": self.wind_direction_rad,
            "dust_opacity": self.dust_opacity,
            "temperature_c": self.temperature_c,
            "battery_budget": self.battery_budget,
            "max_steps": self.max_steps,
            "generation_attempt": self.generation_attempt,
        }
        if include_heightfield:
            data["terrain"] = {
                "n": int(self.terrain_heights.shape[0]),
                "heights": np.round(self.terrain_heights, 4).ravel().tolist(),
            }
        return data


class ScenarioGenerator:
    """Genera terreno, obstáculos, clima y misión a partir de una semilla."""

    def __init__(self, planet_id: str = "mars", *, max_attempts: int = 40):
        self.profile = get_planet(planet_id)
        self.max_attempts = int(max_attempts)

    @staticmethod
    def _rng(seed: int, level: int, attempt: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([int(seed), int(level), int(attempt)]))

    def generate(self, seed: int, difficulty_level: int = 1) -> PlanetaryScenario:
        spec = level_spec(difficulty_level)
        last_reasons: list[str] = []
        for attempt in range(self.max_attempts):
            rng = self._rng(seed, spec.level, attempt)
            try:
                scenario = self._candidate(int(seed), spec, rng, attempt)
            except RuntimeError as exc:
                last_reasons = [str(exc)]
                continue
            valid, last_reasons = self.validate(scenario)
            if valid:
                return scenario
        raise RuntimeError(
            f"no se pudo generar un escenario viable para seed={seed}, nivel={difficulty_level}: "
            + "; ".join(last_reasons)
        )

    def _candidate(
        self,
        seed: int,
        spec: CurriculumLevel,
        rng: np.random.Generator,
        attempt: int,
    ) -> PlanetaryScenario:
        target_type = str(rng.choice(SAMPLE_TYPES))
        requested = int(rng.integers(spec.target_samples_min, spec.target_samples_max + 1))
        capacity = max(requested, min(5, requested + (1 if spec.level >= 4 else 0)))
        relief = spec.relief_m * max(0.45, 1.0 - attempt * 0.018)

        rocks = self._place_rocks(rng, spec)
        samples = self._place_samples(rng, spec, rocks, target_type, requested)
        dunes = self._features(rng, spec, spec.dunes, relief, crater=False)
        craters = self._features(rng, spec, spec.craters, relief, crater=True)
        terrain = self._terrain(dunes, craters, samples, relief, rng)

        traction = float(rng.uniform(spec.traction_min, min(1.0, spec.traction_min + 0.16)))
        traction_zones = self._traction_zones(rng, spec, rocks, samples, traction)
        wind = float(rng.uniform(0.0, spec.wind_max_m_s))
        dust = float(rng.uniform(0.0, spec.dust_max))
        temperature = float(rng.uniform(-85.0, -35.0))
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "seed": seed,
                    "level": spec.level,
                    "attempt": attempt,
                    "rocks": rocks,
                    "samples": [sample.to_dict() for sample in samples],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:10]
        return PlanetaryScenario(
            seed=seed,
            scenario_id=f"mars-L{spec.level}-{fingerprint}",
            generator_version=GENERATOR_VERSION,
            planet=self.profile.to_dict(),
            difficulty_level=spec.level,
            difficulty_name=spec.name,
            active_x_max=spec.active_x_max,
            active_y_half=spec.active_y_half,
            target_type=target_type,
            requested_samples=requested,
            capacity=capacity,
            rocks=rocks,
            samples=samples,
            dunes=dunes,
            craters=craters,
            traction_zones=traction_zones,
            traction=traction,
            wind_speed_m_s=wind,
            wind_direction_rad=float(rng.uniform(-math.pi, math.pi)),
            dust_opacity=dust,
            temperature_c=temperature,
            battery_budget=spec.battery_budget,
            max_steps=spec.max_steps,
            terrain_heights=terrain,
            generation_attempt=attempt,
        )

    def _point(
        self,
        rng: np.random.Generator,
        spec: CurriculumLevel,
        occupied: list[tuple[float, float]],
        *,
        clearance: float,
    ) -> tuple[float, float]:
        for _ in range(300):
            x = float(rng.uniform(2.8, spec.active_x_max - 0.55))
            y = float(rng.uniform(-spec.active_y_half + 0.55, spec.active_y_half - 0.55))
            if _circle_hits_rect(x, y, clearance, BASE) or _circle_hits_rect(x, y, clearance, CRACK):
                continue
            if any(math.hypot(x - px, y - py) < clearance * 2.0 for px, py in occupied):
                continue
            return round(x, 4), round(y, 4)
        raise RuntimeError("no hay espacio para colocar todos los elementos")

    def _place_rocks(self, rng: np.random.Generator, spec: CurriculumLevel) -> list[tuple]:
        rocks: list[tuple] = []
        occupied: list[tuple[float, float]] = []
        for _ in range(spec.rocks):
            x, y = self._point(rng, spec, occupied, clearance=0.72)
            occupied.append((x, y))
            hx, hy = rng.uniform(0.30, 0.62, size=2)
            height = float(rng.uniform(0.35, 0.90))
            rocks.append((x, y, round(float(hx), 4), round(float(hy), 4), round(height, 4)))
        return rocks

    def _place_samples(
        self,
        rng: np.random.Generator,
        spec: CurriculumLevel,
        rocks: list[tuple],
        target_type: str,
        requested: int,
    ) -> list[SampleSpec]:
        occupied = [(rock[0], rock[1]) for rock in rocks]
        samples: list[SampleSpec] = []
        target_total = requested + (1 if spec.level >= 5 else 0)
        for index in range(target_total + spec.distractors):
            x, y = self._point(rng, spec, occupied, clearance=0.58)
            occupied.append((x, y))
            is_target = index < target_total
            if is_target:
                sample_type = target_type
            else:
                sample_type = str(rng.choice([item for item in SAMPLE_TYPES if item != target_type]))
            samples.append(SampleSpec(f"S{index + 1}", sample_type, x, y, is_target))
        rng.shuffle(samples)
        return samples

    def _traction_zones(
        self,
        rng: np.random.Generator,
        spec: CurriculumLevel,
        rocks: list[tuple],
        samples: list[SampleSpec],
        base_traction: float,
    ) -> list[tuple[float, float, float, float, float]]:
        count = 0 if spec.level < 3 else 1 if spec.level < 5 else 2 if spec.level < 7 else 3
        occupied = [(rock[0], rock[1]) for rock in rocks]
        occupied.extend((sample.x, sample.y) for sample in samples)
        zones = []
        for _ in range(count):
            x, y = self._point(rng, spec, occupied, clearance=0.58)
            occupied.append((x, y))
            half_x = float(rng.uniform(0.55, 1.0))
            half_y = float(rng.uniform(0.40, 0.82))
            coefficient = float(rng.uniform(max(0.30, base_traction - 0.28), max(0.31, base_traction - 0.08)))
            zones.append((x, y, round(half_x, 4), round(half_y, 4), round(coefficient, 4)))
        return zones

    def _features(
        self,
        rng: np.random.Generator,
        spec: CurriculumLevel,
        count: int,
        relief: float,
        *,
        crater: bool,
    ) -> list[tuple[float, float, float, float]]:
        features = []
        for _ in range(count):
            x = float(rng.uniform(3.0, spec.active_x_max - 0.4))
            y = float(rng.uniform(-spec.active_y_half + 0.5, spec.active_y_half - 0.5))
            amplitude = float(rng.uniform(0.35, 1.0) * relief)
            if crater:
                amplitude *= -1.0
            sigma = float(rng.uniform(0.65, 1.35))
            features.append((round(x, 4), round(y, 4), round(amplitude, 4), round(sigma, 4)))
        return features

    def _terrain(
        self,
        dunes: list[tuple],
        craters: list[tuple],
        samples: list[SampleSpec],
        relief: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        xs = np.linspace(0.0, MAP_W, TERRAIN_N)
        ys = np.linspace(-MAP_H / 2, MAP_H / 2, TERRAIN_N)
        x_grid, y_grid = np.meshgrid(xs, ys)
        heights = np.zeros_like(x_grid)
        for cx, cy, amplitude, sigma in [*dunes, *craters]:
            heights += amplitude * np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma**2)
            )
        if relief > 0:
            phase_x, phase_y = rng.uniform(0.0, 2.0 * math.pi, size=2)
            heights += relief * 0.12 * np.sin(0.75 * x_grid + phase_x) * np.cos(0.65 * y_grid + phase_y)
        # La base y las zonas de recolección se suavizan; el trayecto entre ellas conserva relieve.
        for px, py, radius in [(BASE[0], BASE[1], 1.35), *[(s.x, s.y, 0.55) for s in samples]]:
            blend = np.clip((np.hypot(x_grid - px, y_grid - py) - radius) / 0.65, 0.0, 1.0)
            heights *= blend
        # La grieta existente sigue siendo una región peligrosa físicamente real.
        cx, cy, hx, hy = CRACK
        heights[(abs(x_grid - cx) <= hx) & (abs(y_grid - cy) <= hy)] = -0.5
        return np.clip(heights, -0.5, 0.38).astype(np.float64)

    def validate(self, scenario: PlanetaryScenario) -> tuple[bool, list[str]]:
        """Comprueba geometría, pendiente, recursos y alcance cinemático aproximado."""
        reasons: list[str] = []
        targets = scenario.target_samples
        if len(targets) < scenario.requested_samples:
            reasons.append("no hay suficientes muestras objetivo")
        if scenario.capacity < scenario.requested_samples:
            reasons.append("la capacidad es menor que la misión")

        reachable = self._reachable_cells(scenario)
        for sample in targets:
            if self._cell(sample.x, sample.y) not in reachable:
                reasons.append(f"muestra inaccesible: {sample.sample_id}")
        route = self._reference_route(scenario) * 1.25  # margen por rodear rocas y pendientes
        minimum_traction = min([scenario.traction, *[zone[4] for zone in scenario.traction_zones]])
        cold_factor = 1.0 + max(0.0, -40.0 - scenario.temperature_c) / 180.0
        traction_factor = 1.0 + max(0.0, 0.75 - minimum_traction) * 0.8
        average_drain = 0.24 * cold_factor * traction_factor
        battery_steps = scenario.battery_budget / max(0.1, average_drain)
        maximum_distance = 0.60 * 0.5 * min(scenario.max_steps, battery_steps) * 0.62
        if route > maximum_distance:
            reasons.append(f"ruta {route:.1f} m supera alcance aproximado {maximum_distance:.1f} m")
        return not reasons, reasons

    @staticmethod
    def _cell(x: float, y: float, cell: float = 0.30) -> tuple[int, int]:
        return int(x / cell), int((y + MAP_H / 2) / cell)

    def _reachable_cells(self, scenario: PlanetaryScenario, cell: float = 0.30) -> set[tuple[int, int]]:
        nx, ny = int(MAP_W / cell), int(MAP_H / cell)
        height = scenario.terrain_heights
        resolution = MAP_W / (height.shape[1] - 1)
        grad_y, grad_x = np.gradient(height, resolution, resolution)
        slopes = np.arctan(np.hypot(grad_x, grad_y))
        wind_margin = min(0.16, scenario.wind_speed_m_s * scenario.planet["atmosphere_density_kg_m3"] * 0.12)

        def free(i: int, j: int) -> bool:
            x = (i + 0.5) * cell
            y = (j + 0.5) * cell - MAP_H / 2
            if x - ROBOT_RADIUS < 0 or x + ROBOT_RADIUS > scenario.active_x_max:
                return False
            if abs(y) + ROBOT_RADIUS > scenario.active_y_half:
                return False
            if _circle_hits_rect(x, y, ROBOT_RADIUS, CRACK):
                return False
            if any(_circle_hits_rect(x, y, ROBOT_RADIUS, rock) for rock in scenario.rocks):
                return False
            row = min(height.shape[0] - 1, max(0, round((y + MAP_H / 2) / resolution)))
            col = min(height.shape[1] - 1, max(0, round(x / resolution)))
            traction = scenario.traction
            for cx, cy, half_x, half_y, zone_traction in scenario.traction_zones:
                if abs(x - cx) <= half_x and abs(y - cy) <= half_y:
                    traction = min(traction, zone_traction)
            friction_limit = math.atan(max(0.05, traction)) * 0.72
            max_slope = max(math.radians(8), friction_limit - wind_margin)
            return bool(slopes[row, col] <= max_slope)

        start = self._cell(BASE[0], BASE[1], cell)
        if not free(*start):
            return set()
        seen = {start}
        queue = [start]
        while queue:
            i, j = queue.pop()
            for candidate in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                a, b = candidate
                if 0 <= a < nx and 0 <= b < ny and candidate not in seen and free(a, b):
                    seen.add(candidate)
                    queue.append(candidate)
        return seen

    @staticmethod
    def _reference_route(scenario: PlanetaryScenario) -> float:
        remaining = [(sample.x, sample.y) for sample in scenario.target_samples]
        current = BASE[:2]
        distance = 0.0
        for _ in range(scenario.requested_samples):
            closest = min(remaining, key=lambda point: math.dist(current, point))
            distance += math.dist(current, closest)
            current = closest
            remaining.remove(closest)
        return distance + math.dist(current, BASE[:2])
