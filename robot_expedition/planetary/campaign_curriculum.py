"""Promotion from independent frozen-policy evaluations, never training returns."""

import json
from pathlib import Path

from .campaign import SCENARIO_IDS


class CampaignCurriculum:
    def __init__(
        self,
        state_path=None,
        initial_level=1,
        minimum_episodes=50,
        success_threshold=0.85,
        max_collisions=2.0,
        rehearsal_probability=0.2,
    ):
        self.state_path = Path(state_path) if state_path else None
        self.current_level = initial_level
        self.minimum_episodes = minimum_episodes
        self.success_threshold = success_threshold
        self.max_collisions = max_collisions
        self.rehearsal_probability = rehearsal_probability
        self.mastered = []
        self.evaluations = {}
        self._load()
        if self.state_path and not self.state_path.exists():
            self._save()

    def _load(self):
        if self.state_path and self.state_path.exists():
            d = json.loads(self.state_path.read_text())
            if d.get("schema") != "fixed-campaign-curriculum/v1":
                raise ValueError(
                    "use a separate campaign curriculum state; legacy state is incompatible"
                )
            self.current_level = d["current_level"]
            self.mastered = d["mastered"]
            self.evaluations = d["evaluations"]

    def _save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=2))
            tmp.replace(self.state_path)

    def get_current_scenario(self):
        self._load()
        return SCENARIO_IDS[self.current_level - 1]

    def select_level(self, rng):
        self._load()
        return (
            int(rng.integers(1, self.current_level))
            if self.current_level > 1 and rng.random() < self.rehearsal_probability
            else self.current_level
        )

    def record_result(self, success, *, evaluated_level):
        return False  # training must not promote the policy

    def evaluate_scenario_mastery(self, result):
        self._load()
        cases = result["cases"]
        level = int(result["scenario_level"])
        if result.get("evaluation_kind") != "mastery":
            raise ValueError("only mastery evaluations can change official progression")
        if any(c["difficulty_level"] != level for c in cases):
            raise ValueError("mixed scenarios cannot prove scenario mastery")
        seeds = {c["seed"] for c in cases}
        passed = (
            len(cases) >= self.minimum_episodes
            and len(seeds) == len(cases)
            and sum(bool(c["success"]) for c in cases) / len(cases)
            >= self.success_threshold
            and sum(c["collisions"] for c in cases) / len(cases) <= self.max_collisions
            and all(
                not c["success"]
                or (
                    c["samples_collected"] >= c["requested_samples"]
                    and c["battery"] > 0
                )
                for c in cases
            )
        )
        self.evaluations[str(level)] = {
            "passed": passed,
            "episodes": len(cases),
            "success_rate": sum(bool(c["success"]) for c in cases) / max(1, len(cases)),
            "policy_version": result.get("policy_version"),
            "training_step": result.get("training_step"),
        }
        self._save()
        return passed

    def advance_scenario(self):
        if not self.evaluations.get(str(self.current_level), {}).get("passed"):
            return False
        if self.current_level not in self.mastered:
            self.mastered.append(self.current_level)
        if self.current_level < 10:
            self.current_level += 1
        self._save()
        return True

    def return_to_previous_scenario(self, level=None):
        self.current_level = max(
            1, int(level if level is not None else self.current_level - 1)
        )
        self.mastered = [n for n in self.mastered if n < self.current_level]
        self._save()

    def snapshot(self):
        return {
            "schema": "fixed-campaign-curriculum/v1",
            "current_level": self.current_level,
            "current_scenario": SCENARIO_IDS[self.current_level - 1],
            "mastered": self.mastered,
            "pending": [
                sid for i, sid in enumerate(SCENARIO_IDS, 1) if i not in self.mastered
            ],
            "evaluations": self.evaluations,
            "settings": {
                "minimum_episodes": self.minimum_episodes,
                "success_threshold": self.success_threshold,
                "max_collisions": self.max_collisions,
            },
        }

    get_curriculum_progress = snapshot
