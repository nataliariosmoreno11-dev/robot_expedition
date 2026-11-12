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


SCHEMA_VERSION = 1
DEFAULT_DB = Path(os.environ.get("EXPEDITION_EPISODE_DB", "logs/planetary_episodes.sqlite"))


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
    """Almacén transaccional; cada paso se confirma sin acumular episodios en RAM."""

    def __init__(self, path: str | Path = DEFAULT_DB):
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
                """
            )
            existing = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing and int(existing["value"]) != SCHEMA_VERSION:
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
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO episodes(
                    episode_uid, schema_version, run_id, policy_version, mode, seed,
                    scenario_id, planet, difficulty_level, difficulty_name,
                    target_type, requested_samples, scenario_blob, map_blob, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
