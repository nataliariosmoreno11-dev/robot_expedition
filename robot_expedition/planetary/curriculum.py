"""Curriculum learning basado en rendimiento y con repaso de niveles previos."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CurriculumLevel:
    level: int
    name: str
    active_x_max: float
    active_y_half: float
    target_samples_min: int
    target_samples_max: int
    distractors: int
    rocks: int
    dunes: int
    craters: int
    relief_m: float
    wind_max_m_s: float
    dust_max: float
    traction_min: float
    battery_budget: float
    max_steps: int


LEVELS = (
    CurriculumLevel(1, "llanura inicial", 5.0, 2.5, 1, 1, 0, 1, 0, 0, 0.02, 2.0, 0.02, 0.95, 100.0, 220),
    CurriculumLevel(2, "rocas dispersas", 6.0, 3.0, 1, 2, 1, 2, 1, 0, 0.05, 4.0, 0.05, 0.90, 100.0, 260),
    CurriculumLevel(3, "pendientes y fricción", 7.0, 3.5, 2, 2, 1, 3, 2, 0, 0.10, 6.0, 0.10, 0.78, 98.0, 300),
    CurriculumLevel(4, "cráteres y viento", 8.0, 4.0, 2, 3, 1, 4, 2, 1, 0.15, 10.0, 0.18, 0.72, 96.0, 340),
    CurriculumLevel(5, "dunas y polvo", 9.0, 4.5, 2, 3, 2, 6, 3, 1, 0.20, 14.0, 0.30, 0.66, 92.0, 380),
    CurriculumLevel(6, "rutas extensas", 10.0, 5.0, 3, 4, 2, 8, 3, 2, 0.24, 18.0, 0.38, 0.62, 88.0, 430),
    CurriculumLevel(7, "meteorología variable", 11.0, 5.3, 3, 4, 3, 10, 4, 2, 0.27, 22.0, 0.48, 0.58, 84.0, 480),
    CurriculumLevel(8, "expedición marciana", 11.4, 5.4, 4, 4, 3, 12, 4, 3, 0.30, 26.0, 0.58, 0.54, 80.0, 540),
)


def level_spec(level: int) -> CurriculumLevel:
    if not 1 <= int(level) <= len(LEVELS):
        raise ValueError(f"difficulty_level debe estar entre 1 y {len(LEVELS)}")
    return LEVELS[int(level) - 1]


class CurriculumManager:
    """Avanza por éxito reciente y mezcla niveles anteriores para evitar olvido."""

    def __init__(
        self,
        state_path: str | Path | None = None,
        *,
        initial_level: int = 1,
        window_size: int = 20,
        minimum_episodes: int = 10,
        success_threshold: float = 0.75,
        rehearsal_probability: float = 0.20,
    ):
        self.state_path = Path(state_path) if state_path else None
        self.window_size = int(window_size)
        self.minimum_episodes = int(minimum_episodes)
        self.success_threshold = float(success_threshold)
        self.rehearsal_probability = float(rehearsal_probability)
        self.current_level = level_spec(initial_level).level
        self.outcomes: deque[bool] = deque(maxlen=self.window_size)
        self.total_episodes = 0
        self._load()

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("versión de curriculum.json no soportada")
        self.current_level = level_spec(int(data["current_level"])).level
        self.total_episodes = int(data.get("total_episodes", 0))
        self.outcomes = deque((bool(v) for v in data.get("outcomes", [])), maxlen=self.window_size)

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "current_level": self.current_level,
            "total_episodes": self.total_episodes,
            "outcomes": list(self.outcomes),
            "settings": {
                "window_size": self.window_size,
                "minimum_episodes": self.minimum_episodes,
                "success_threshold": self.success_threshold,
                "rehearsal_probability": self.rehearsal_probability,
            },
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    @property
    def success_rate(self) -> float | None:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else None

    def select_level(self, rng) -> int:
        """Usa el nivel actual; ocasionalmente practica uno anterior."""
        self._load()
        if self.current_level > 1 and float(rng.random()) < self.rehearsal_probability:
            return int(rng.integers(1, self.current_level))
        return self.current_level

    def record_result(self, success: bool, *, evaluated_level: int) -> bool:
        """Registra el nivel actual y devuelve True cuando avanza el currículo."""
        self._load()
        self.total_episodes += 1
        advanced = False
        if int(evaluated_level) == self.current_level:
            self.outcomes.append(bool(success))
            if (
                len(self.outcomes) >= self.minimum_episodes
                and self.success_rate is not None
                and self.success_rate >= self.success_threshold
                and self.current_level < len(LEVELS)
            ):
                self.current_level += 1
                self.outcomes.clear()
                advanced = True
        self._save()
        return advanced

    def snapshot(self) -> dict:
        spec = level_spec(self.current_level)
        return {
            "schema_version": SCHEMA_VERSION,
            "current_level": self.current_level,
            "level_name": spec.name,
            "success_rate": self.success_rate,
            "window_episodes": len(self.outcomes),
            "total_episodes": self.total_episodes,
            "settings": {
                "success_threshold": self.success_threshold,
                "minimum_episodes": self.minimum_episodes,
                "rehearsal_probability": self.rehearsal_probability,
            },
            "level": asdict(spec),
        }
