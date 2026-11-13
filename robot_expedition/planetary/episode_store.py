"""SQLite incremental para registrar y reproducir cada intento completo."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
DEFAULT_DB = Path(os.environ.get("EXPEDITION_EPISODE_DB", "logs/planetary_episodes.sqlite"))
DEFAULT_MAX_EXPERIMENTS = int(os.environ.get("EXPEDITION_MAX_EXPERIMENTS", "10"))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _pack(value: Any) -> bytes:
    return zlib.compress(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), 6)


def _unpack(value: bytes | None) -> Any:
    return None if value is None else json.loads(zlib.decompress(value).decode("utf-8"))


class EpisodeWriter:
    def __init__(self, store: "EpisodeStore", episode_number: int, episode_uid: str):
        self.store = store
        self.episode_number = int(episode_number)
        self.episode_uid = episode_uid
        self.step_index = -1
        self.closed = False

    def record_step(
        self,
        *,
        state: dict,
        action: int,
        action_name: str,
        reward: float,
        cumulative_reward: float,
        event: str | None,
        collision: bool,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.closed:
            raise RuntimeError("el episodio ya fue cerrado")
        self.step_index += 1
        self.store._record_step(
            self.episode_number,
            self.step_index,
            state,
            action,
            action_name,
            reward,
            cumulative_reward,
            event,
            collision,
            terminated,
            truncated,
        )

    def finish(
        self,
        *,
        end: str,
        success: bool,
        cumulative_reward: float,
        duration_s: float,
        steps: int,
        samples_collected: int,
        collisions: int,
        path_length: float,
    ) -> None:
        if self.closed:
            return
        self.store._finish_episode(
            self.episode_number,
            end=end,
            success=success,
            cumulative_reward=cumulative_reward,
            duration_s=duration_s,
            steps=steps,
            samples_collected=samples_collected,
            collisions=collisions,
            path_length=path_length,
        )
        self.closed = True


class EpisodeStore:
    """Métricas de todos los intentos y un banco acotado de replays completos.

    Las filas creadas con el esquema anterior se consideran históricas y nunca se
    eliminan automáticamente. El límite solo se aplica a experimentos nuevos
    gestionados por retención.
    """

    def __init__(self, path: str | Path = DEFAULT_DB, *, max_experiments: int = DEFAULT_MAX_EXPERIMENTS):
        if int(max_experiments) < 1:
            raise ValueError("max_experiments debe ser al menos 1")
        self.max_experiments = int(max_experiments)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_uid TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    scenario_id TEXT NOT NULL,
                    planet TEXT NOT NULL,
                    difficulty_level INTEGER NOT NULL,
                    difficulty_name TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    requested_samples INTEGER NOT NULL,
                    scenario_blob BLOB NOT NULL,
                    map_blob BLOB NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    end_cause TEXT,
                    success INTEGER,
                    cumulative_reward REAL NOT NULL DEFAULT 0,
                    duration_s REAL,
                    steps INTEGER NOT NULL DEFAULT 0,
                    samples_collected INTEGER NOT NULL DEFAULT 0,
                    collisions INTEGER NOT NULL DEFAULT 0,
                    path_length REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS episode_steps (
                    episode_number INTEGER NOT NULL,
                    step_index INTEGER NOT NULL,
                    state_blob BLOB NOT NULL,
                    action INTEGER NOT NULL,
                    action_name TEXT NOT NULL,
                    reward REAL NOT NULL,
                    cumulative_reward REAL NOT NULL,
                    event TEXT,
                    collision INTEGER NOT NULL,
                    terminated INTEGER NOT NULL,
                    truncated INTEGER NOT NULL,
                    PRIMARY KEY (episode_number, step_index),
                    FOREIGN KEY (episode_number) REFERENCES episodes(episode_number) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id, episode_number);
                CREATE INDEX IF NOT EXISTS idx_episodes_scenario ON episodes(scenario_id);
                CREATE TABLE IF NOT EXISTS training_metrics (
                    episode_number INTEGER PRIMARY KEY,
                    episode_uid TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    scenario_id TEXT NOT NULL,
                    planet TEXT NOT NULL,
                    difficulty_level INTEGER NOT NULL,
                    terrain_signature TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    cumulative_reward REAL NOT NULL,
                    samples_collected INTEGER NOT NULL,
                    requested_samples INTEGER NOT NULL,
                    collisions INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    path_length REAL NOT NULL,
                    end_cause TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    replay_retained INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_run ON training_metrics(run_id, episode_number);
                CREATE INDEX IF NOT EXISTS idx_metrics_level ON training_metrics(difficulty_level);
                """
            )
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(episodes)")}
            migrations = {
                "retention_managed": "INTEGER NOT NULL DEFAULT 0",
                "protected": "INTEGER NOT NULL DEFAULT 0",
                "selection_reason": "TEXT NOT NULL DEFAULT 'histórico'",
                "terrain_signature": "TEXT NOT NULL DEFAULT 'legado'",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    self._connection.execute(f"ALTER TABLE episodes ADD COLUMN {name} {declaration}")
            # Resumir históricos es aditivo: conserva intactos sus blobs y pasos.
            self._connection.execute(
                """
                INSERT OR IGNORE INTO training_metrics(
                    episode_number, episode_uid, run_id, seed, scenario_id, planet,
                    difficulty_level, terrain_signature, success, cumulative_reward,
                    samples_collected, requested_samples, collisions, steps, path_length,
                    end_cause, completed_at, replay_retained
                ) SELECT episode_number, episode_uid, run_id, seed, scenario_id, planet,
                    difficulty_level, terrain_signature, success, cumulative_reward,
                    samples_collected, requested_samples, collisions, steps, path_length,
                    end_cause, ended_at, 1
                FROM episodes WHERE ended_at IS NOT NULL AND success IS NOT NULL
                """
            )
            existing = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing and int(existing["value"]) > SCHEMA_VERSION:
                raise ValueError("versión de SQLite no soportada")
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def start_episode(
        self,
        *,
        run_id: str,
        policy_version: str,
        mode: str,
        seed: int,
        scenario: dict,
        map_spec: dict,
    ) -> EpisodeWriter:
        episode_uid = uuid.uuid4().hex
        terrain_signature = self._terrain_signature(scenario)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO episodes(
                    episode_uid, schema_version, run_id, policy_version, mode, seed,
                    scenario_id, planet, difficulty_level, difficulty_name,
                    target_type, requested_samples, scenario_blob, map_blob, started_at,
                    retention_managed, protected, selection_reason, terrain_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
                """,
                (
                    episode_uid,
                    SCHEMA_VERSION,
                    run_id,
                    policy_version,
                    mode,
                    int(seed),
                    scenario["scenario_id"],
                    scenario["planet"]["planet_id"],
                    int(scenario["difficulty_level"]),
                    scenario["difficulty_name"],
                    scenario["target_type"],
                    int(scenario["requested_samples"]),
                    _pack(scenario),
                    _pack(map_spec),
                    _now(),
                    "candidato en curso",
                    terrain_signature,
                ),
            )
        return EpisodeWriter(self, int(cursor.lastrowid), episode_uid)

    def _record_step(
        self,
        episode_number: int,
        step_index: int,
        state: dict,
        action: int,
        action_name: str,
        reward: float,
        cumulative_reward: float,
        event: str | None,
        collision: bool,
        terminated: bool,
        truncated: bool,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO episode_steps(
                    episode_number, step_index, state_blob, action, action_name,
                    reward, cumulative_reward, event, collision, terminated, truncated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_number,
                    step_index,
                    _pack(state),
                    int(action),
                    action_name,
                    float(reward),
                    float(cumulative_reward),
                    event,
                    int(collision),
                    int(terminated),
                    int(truncated),
                ),
            )

    def _finish_episode(self, episode_number: int, **result: Any) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE episodes SET ended_at=?, end_cause=?, success=?, cumulative_reward=?,
                    duration_s=?, steps=?, samples_collected=?, collisions=?, path_length=?
                WHERE episode_number=?
                """,
                (
                    _now(),
                    result["end"],
                    int(result["success"]),
                    float(result["cumulative_reward"]),
                    float(result["duration_s"]),
                    int(result["steps"]),
                    int(result["samples_collected"]),
                    int(result["collisions"]),
                    float(result["path_length"]),
                    episode_number,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_number=?", (episode_number,)
            ).fetchone()
            self._connection.execute(
                """
                INSERT OR REPLACE INTO training_metrics(
                    episode_number, episode_uid, run_id, seed, scenario_id, planet,
                    difficulty_level, terrain_signature, success, cumulative_reward,
                    samples_collected, requested_samples, collisions, steps, path_length,
                    end_cause, completed_at, replay_retained
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    row["episode_number"], row["episode_uid"], row["run_id"], row["seed"],
                    row["scenario_id"], row["planet"], row["difficulty_level"],
                    row["terrain_signature"], int(result["success"]),
                    float(result["cumulative_reward"]), int(result["samples_collected"]),
                    row["requested_samples"], int(result["collisions"]), int(result["steps"]),
                    float(result["path_length"]), result["end"], _now(),
                ),
            )
            self._rebalance_retained()

    @staticmethod
    def _terrain_signature(scenario: dict) -> str:
        counts = {
            "cráteres": len(scenario.get("craters", [])),
            "dunas": len(scenario.get("dunes", [])),
            "rocas": len(scenario.get("rocks", [])),
            "suelo_suelto": len(scenario.get("traction_zones", [])),
        }
        dominant = max(counts, key=lambda key: (counts[key], key))
        return f"{dominant}:{counts[dominant]}"

    @staticmethod
    def _feature(row: sqlite3.Row) -> tuple[float, ...]:
        scenario = _unpack(row["scenario_blob"])
        return (
            float(row["difficulty_level"]) / 8.0,
            float(bool(row["success"])),
            min(len(scenario.get("rocks", [])) / 12.0, 1.0),
            min(len(scenario.get("dunes", [])) / 6.0, 1.0),
            min(len(scenario.get("craters", [])) / 6.0, 1.0),
            min(len(scenario.get("traction_zones", [])) / 3.0, 1.0),
            min(float(scenario.get("wind_speed_m_s", 0.0)) / 30.0, 1.0),
            min(float(scenario.get("dust_opacity", 0.0)), 1.0),
        )

    @staticmethod
    def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5

    def _rebalance_retained(self) -> None:
        """Descarta el replay más redundante; nunca toca históricos ni protegidos."""
        while True:
            rows = self._connection.execute(
                "SELECT * FROM episodes WHERE retention_managed=1 AND ended_at IS NOT NULL ORDER BY episode_number"
            ).fetchall()
            if len(rows) <= self.max_experiments:
                break
            removable = [row for row in rows if not row["protected"]]
            if not removable:
                break
            features = {row["episode_number"]: self._feature(row) for row in rows}
            def redundancy(row: sqlite3.Row) -> tuple[float, int]:
                others = [other for other in rows if other["episode_number"] != row["episode_number"]]
                nearest = min(
                    (self._distance(features[row["episode_number"]], features[other["episode_number"]]) for other in others),
                    default=float("inf"),
                )
                return nearest, row["episode_number"]
            victim = min(removable, key=redundancy)
            number = int(victim["episode_number"])
            self._connection.execute(
                "UPDATE training_metrics SET replay_retained=0 WHERE episode_number=?", (number,)
            )
            self._connection.execute("DELETE FROM episodes WHERE episode_number=?", (number,))

        retained = self._connection.execute(
            "SELECT * FROM episodes WHERE retention_managed=1 AND ended_at IS NOT NULL"
        ).fetchall()
        for row in retained:
            reason = "protegido manualmente" if row["protected"] else (
                f"selección diversa · L{row['difficulty_level']} · {row['terrain_signature']} · "
                f"{'éxito' if row['success'] else 'fallo'}"
            )
            self._connection.execute(
                "UPDATE episodes SET selection_reason=? WHERE episode_number=?",
                (reason, row["episode_number"]),
            )

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict:
        return {
            "episode_number": row["episode_number"],
            "episode_uid": row["episode_uid"],
            "run_id": row["run_id"],
            "policy_version": row["policy_version"],
            "mode": row["mode"],
            "seed": row["seed"],
            "scenario_id": row["scenario_id"],
            "planet": row["planet"],
            "difficulty_level": row["difficulty_level"],
            "difficulty_name": row["difficulty_name"],
            "target_type": row["target_type"],
            "requested_samples": row["requested_samples"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "end": row["end_cause"] or "en_progreso",
            "success": None if row["success"] is None else bool(row["success"]),
            "reward": row["cumulative_reward"],
            "duration_s": row["duration_s"],
            "steps": row["steps"],
            "samples_collected": row["samples_collected"],
            "collisions": row["collisions"],
            "path_length": row["path_length"],
            "protected": bool(row["protected"]),
            "retention_managed": bool(row["retention_managed"]),
            "selection_reason": row["selection_reason"],
            "terrain_signature": row["terrain_signature"],
        }

    def storage_status(self) -> dict:
        managed = self._connection.execute(
            "SELECT COUNT(*) AS total, SUM(protected) AS protected FROM episodes "
            "WHERE retention_managed=1 AND ended_at IS NOT NULL"
        ).fetchone()
        historical = self._connection.execute(
            "SELECT COUNT(*) AS total FROM episodes WHERE retention_managed=0"
        ).fetchone()["total"]
        metric_count = self._connection.execute("SELECT COUNT(*) AS total FROM training_metrics").fetchone()["total"]
        return {
            "max_experiments": self.max_experiments,
            "retained_experiments": managed["total"],
            "protected_experiments": managed["protected"] or 0,
            "historical_untouched": historical,
            "summarized_episodes": metric_count,
        }

    def protect_episode(self, episode_number: int, protected: bool = True) -> dict:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_number=?", (int(episode_number),)
            ).fetchone()
            if row is None:
                raise KeyError(f"no existe el episodio {episode_number}")
            if protected and row["retention_managed"]:
                count = self._connection.execute(
                    "SELECT COUNT(*) AS total FROM episodes WHERE retention_managed=1 AND protected=1"
                ).fetchone()["total"]
                if count >= self.max_experiments and not row["protected"]:
                    raise ValueError("no se pueden proteger más experimentos que el límite configurado")
            self._connection.execute(
                "UPDATE episodes SET protected=?, selection_reason=? WHERE episode_number=?",
                (int(protected), "protegido manualmente" if protected else "selección automática", int(episode_number)),
            )
            self._rebalance_retained()
            updated = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_number=?", (int(episode_number),)
            ).fetchone()
            return self._summary(updated)

    def metrics(self, *, run_id: str | None = None) -> dict:
        where, params = (" WHERE run_id=?", [run_id]) if run_id else ("", [])
        rows = self._connection.execute(
            f"SELECT * FROM training_metrics{where} ORDER BY episode_number", params
        ).fetchall()
        if not rows:
            return {"episodes": 0, "success_rate": 0.0, "reward_mean": 0.0,
                    "samples_collected": 0, "by_difficulty": {}, "evolution": []}
        by_level: dict[str, dict] = {}
        for row in rows:
            bucket = by_level.setdefault(str(row["difficulty_level"]), {"episodes": 0, "successes": 0, "reward": 0.0, "samples": 0})
            bucket["episodes"] += 1
            bucket["successes"] += row["success"]
            bucket["reward"] += row["cumulative_reward"]
            bucket["samples"] += row["samples_collected"]
        for bucket in by_level.values():
            bucket["success_rate"] = round(bucket.pop("successes") / bucket["episodes"], 4)
            bucket["reward_mean"] = round(bucket.pop("reward") / bucket["episodes"], 4)
        size = max(1, min(100, len(rows) // 20 or 1))
        evolution = []
        for start in range(0, len(rows), size):
            group = rows[start:start + size]
            evolution.append({
                "episode": group[-1]["episode_number"],
                "reward_mean": round(sum(row["cumulative_reward"] for row in group) / len(group), 4),
                "success_rate": round(sum(row["success"] for row in group) / len(group), 4),
                "difficulty_mean": round(sum(row["difficulty_level"] for row in group) / len(group), 3),
                "samples_collected": sum(row["samples_collected"] for row in group),
            })
        return {
            "episodes": len(rows),
            "success_rate": round(sum(row["success"] for row in rows) / len(rows), 4),
            "reward_mean": round(sum(row["cumulative_reward"] for row in rows) / len(rows), 4),
            "samples_collected": sum(row["samples_collected"] for row in rows),
            "by_difficulty": by_level,
            "evolution": evolution,
        }

    def list_episodes(self, *, limit: int = 100, offset: int = 0, run_id: str | None = None) -> list[dict]:
        where, params = "", []
        if run_id:
            where, params = "WHERE run_id=?", [run_id]
        rows = self._connection.execute(
            f"SELECT * FROM episodes {where} ORDER BY episode_number DESC LIMIT ? OFFSET ?",
            [*params, min(max(int(limit), 1), 1000), max(int(offset), 0)],
        ).fetchall()
        return [self._summary(row) for row in rows]

    def get_episode(self, episode_number: int, *, include_steps: bool = True) -> dict:
        row = self._connection.execute(
            "SELECT * FROM episodes WHERE episode_number=?", (int(episode_number),)
        ).fetchone()
        if row is None:
            raise KeyError(f"no existe el episodio {episode_number}")
        result = {
            **self._summary(row),
            "schema_version": row["schema_version"],
            "scenario": _unpack(row["scenario_blob"]),
            "map": _unpack(row["map_blob"]),
        }
        if include_steps:
            steps = self._connection.execute(
                "SELECT * FROM episode_steps WHERE episode_number=? ORDER BY step_index",
                (int(episode_number),),
            ).fetchall()
            result["timeline"] = [
                {
                    "step": item["step_index"],
                    "state": _unpack(item["state_blob"]),
                    "action": item["action"],
                    "action_name": item["action_name"],
                    "reward": item["reward"],
                    "cumulative_reward": item["cumulative_reward"],
                    "event": item["event"],
                    "collision": bool(item["collision"]),
                    "terminated": bool(item["terminated"]),
                    "truncated": bool(item["truncated"]),
                }
                for item in steps
            ]
        return result

    def load_scenario(self, episode_number: int) -> dict:
        return self.get_episode(episode_number, include_steps=False)["scenario"]

    def replay_episode(self, episode_number: int) -> dict:
        episode = self.get_episode(episode_number, include_steps=True)
        timeline = episode["timeline"]
        return {
            "map": episode["map"],
            "episode": {
                "id": f"episode-{episode_number}",
                "mode": "entrenamiento-replay",
                "episode": episode_number,
                "run_id": episode["run_id"],
                "model_name": episode["policy_version"],
                "environment_version": episode["map"].get("environment_version", "planetary-exploration/1.0.0"),
                "level_id": f"L{episode['difficulty_level']}",
                "scenario_id": episode["scenario_id"],
                "difficulty": f"L{episode['difficulty_level']} · {episode['difficulty_name']}",
                "planet": episode["planet"],
                "objective": f"Recolectar {episode['requested_samples']} × {episode['target_type']}",
                "requested_samples": episode["requested_samples"],
                "samples_collected": episode["samples_collected"],
                "reward": episode["reward"],
                "steps": episode["steps"],
                "success": episode["success"],
                "end": episode["end"],
                "collisions": episode["collisions"],
                "path_length": episode["path_length"],
                "frames": [item["state"]["frame"] for item in timeline],
                "mission_timeline": [item["state"].get("mission", {}) for item in timeline],
                "actions_timeline": [item["action_name"] for item in timeline],
            },
        }

    def compare_episodes(self, *episode_numbers: int) -> dict:
        if len(episode_numbers) < 2:
            raise ValueError("compare_episodes requiere al menos dos números")
        episodes = []
        for number in episode_numbers:
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_number=?", (int(number),)
            ).fetchone()
            if row is None:
                raise KeyError(f"no existe el episodio {number}")
            episodes.append(self._summary(row))
        shortest_success = min(
            (item for item in episodes if item["success"]),
            key=lambda item: item["steps"],
            default=None,
        )
        return {
            "episodes": episodes,
            "best_reward": max(episodes, key=lambda item: item["reward"])["episode_number"],
            "shortest_success": (
                shortest_success["episode_number"] if shortest_success is not None else None
            ),
        }

    def close(self) -> None:
        self._connection.close()


def show_episode_data(episode_number: int, path: str | Path = DEFAULT_DB) -> dict:
    store = EpisodeStore(path)
    try:
        return store.get_episode(episode_number)
    finally:
        store.close()


def replay_episode(episode_number: int, path: str | Path = DEFAULT_DB) -> dict:
    store = EpisodeStore(path)
    try:
        return store.replay_episode(episode_number)
    finally:
        store.close()


def load_scenario(episode_number: int, path: str | Path = DEFAULT_DB) -> dict:
    store = EpisodeStore(path)
    try:
        return store.load_scenario(episode_number)
    finally:
        store.close()


def compare_episodes(*episode_numbers: int, path: str | Path = DEFAULT_DB) -> dict:
    store = EpisodeStore(path)
    try:
        return store.compare_episodes(*episode_numbers)
    finally:
        store.close()
