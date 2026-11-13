"""Ten immutable, editable missions. Sensor seeds never select map geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .profiles import get_planet
from .scenario import PlanetaryScenario, SampleSpec, ScenarioGenerator

CATALOG = Path(__file__).resolve().parents[2] / "configs" / "campaign"
SCENARIO_IDS = tuple(f"scenario_{i:02d}" for i in range(1, 11))


def load_scenario(scenario_id: str) -> PlanetaryScenario:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unknown campaign scenario: {scenario_id}")
    definition = json.loads((CATALOG / f"{scenario_id}.json").read_text())
    if definition["dimensions"] != [12, 12]:
        raise ValueError(
            "the current physical world requires dimensions [12,12] metres"
        )
    samples = [SampleSpec(**sample) for sample in definition["samples"]]
    if definition["scenario_id"] != scenario_id:
        raise ValueError("scenario identifier does not match its filename")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("sample identifiers must be unique")
    targets = sum(s.target for s in samples)
    if not 1 <= targets == definition["requested_samples"] <= definition["capacity"]:
        raise ValueError(
            "mission must have exactly the requested targets and sufficient capacity"
        )
    if not (
        0 < definition["battery_budget"] <= 100
        and definition["max_steps"] > 0
        and definition["energy_scale"] > 0
    ):
        raise ValueError("invalid mission resource budget")
    terrain = ScenarioGenerator()._terrain(
        definition["dunes"],
        definition["craters"],
        samples,
        definition["relief_m"],
        np.random.default_rng(definition["terrain_seed"]),
    )
    return PlanetaryScenario(
        seed=definition["terrain_seed"],
        scenario_id=scenario_id,
        generator_version="fixed-campaign/2.0.0",
        planet=get_planet("mars").to_dict(),
        difficulty_level=int(scenario_id[-2:]),
        difficulty_name=definition["name"],
        active_x_max=11.8,
        active_y_half=5.8,
        target_type=definition["target_type"],
        requested_samples=definition["requested_samples"],
        capacity=definition["capacity"],
        rocks=[tuple(r) for r in definition["rocks"]],
        samples=samples,
        dunes=definition["dunes"],
        craters=definition["craters"],
        traction_zones=definition["traction_zones"],
        traction=definition["traction"],
        wind_speed_m_s=definition["wind_speed_m_s"],
        wind_direction_rad=definition["wind_direction_rad"],
        dust_opacity=definition["dust_opacity"],
        temperature_c=definition["temperature_c"],
        battery_budget=definition["battery_budget"],
        max_steps=definition["max_steps"],
        terrain_heights=terrain,
        energy_scale=definition["energy_scale"],
        start=tuple(definition["start"]),
    )


def catalog_hash() -> str:
    return hashlib.sha256(
        b"".join((CATALOG / f"{sid}.json").read_bytes() for sid in SCENARIO_IDS)
    ).hexdigest()


class FixedScenarioGenerator:
    def generate(self, seed: int, difficulty_level: int = 1) -> PlanetaryScenario:
        return load_scenario(f"scenario_{difficulty_level:02d}")
