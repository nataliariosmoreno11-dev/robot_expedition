"""Suite de evaluacion fija de Robot Expedition (paso 2 del plan).

Evalua cualquier politica congelada policy(obs) -> accion con semillas
versionadas (configs/suites) y devuelve resultados por caso y metricas de
mision. La recompensa acumulada se reporta aparte de la tasa de exito: no son
lo mismo.

    python pipelines/evaluate.py --policy rule --level n2
    python pipelines/evaluate.py --policy qlearning --model models/q.npz --level n2
    python pipelines/evaluate.py --policy ppo --model model.zip --vecnormalize vecnormalize.pkl --level n2
    python pipelines/evaluate.py --baselines          # aleatorio y regla en N1, N2 y N3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_expedition.envs.beacon_expedition import (  # noqa: E402
    BASE, BEACONS, ENVIRONMENT_VERSION, BeaconExpeditionEnv, environment_hash, load_level_config,
)
from pipelines.policies import RandomPolicy, RulePolicy, make_policy  # noqa: E402

SUITES_DIR = ROOT / "configs" / "suites"
RESULTS_DIR = ROOT / "docs" / "resultados"
RESERVED_SEEDS_ENV = "EXPEDITION_TEST_SEEDS"
END_CAUSES = ("exito", "grieta", "volco", "bateria", "tiempo")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_commit() -> str:
    return _git("rev-parse", "HEAD") or "desconocido"


def _git_dirty() -> bool | None:
    """True si el codigo evaluado tiene cambios sin commit (el commit no lo identifica del todo)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def load_suite(suite_id: str) -> dict:
    """Lee configs/suites/<suite_id>.yaml y resuelve sus semillas.

    El test reservado (seeds_source: reserved) no guarda semillas en el repo:
    se leen de la variable EXPEDITION_TEST_SEEDS ("101,202,...") o de
    configs/suites/<suite_id>.local.yaml (ignorado por Git, clave seeds).
    """
    path = SUITES_DIR / f"{suite_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no existe la suite {suite_id} ({path})")
    raw = path.read_text(encoding="utf-8")
    suite = yaml.safe_load(raw)
    suite["suite_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if suite.get("seeds_source") == "reserved":
        seeds = None
        if os.environ.get(RESERVED_SEEDS_ENV):
            seeds = [int(s) for s in os.environ[RESERVED_SEEDS_ENV].split(",") if s.strip()]
        else:
            local = SUITES_DIR / f"{suite_id}.local.yaml"
            if local.exists():
                seeds = [int(s) for s in yaml.safe_load(local.read_text(encoding="utf-8"))["seeds"]]
        if not seeds:
            raise RuntimeError(f"la suite {suite_id} es el test reservado: define {RESERVED_SEEDS_ENV} "
                               f"o crea {SUITES_DIR / (suite_id + '.local.yaml')} (no se sube a Git)")
        suite["seeds"] = seeds
    else:
        suite["seeds"] = list(range(int(suite["seed_start"]), int(suite["seed_start"]) + int(suite["episodes"])))
    return suite


def reference_route(beacon_count: int) -> float:
    """Ruta de referencia de centro a centro: base -> balizas en orden -> base (m)."""
    points = [BASE[:2], *BEACONS[:beacon_count], BASE[:2]]
    return float(sum(math.dist(a, b) for a, b in zip(points, points[1:])))


def frame(env: BeaconExpeditionEnv, action: int, reward: float, collision: bool) -> list:
    """Cuadro del contrato: [x, y, z, qw, qx, qy, qz, etapa, bateria, accion, recompensa, choque, rayos]."""
    return [round(env.x, 3), round(env.y, 3), round(env.z, 3), *np.round(env.quat, 4).tolist(),
            env.stage, round(env.battery, 2), int(action), round(float(reward), 3), int(collision),
            [round(float(r), 2) for r in env.last_rays]]


def run_case(env: BeaconExpeditionEnv, policy, seed: int, frames: list | None = None) -> dict:
    """Un episodio completo con la politica congelada. Si se pasa frames, graba los cuadros."""
    if hasattr(policy, "reset"):
        policy.reset(seed)
    obs, info = env.reset(seed=seed)
    reference_route_m = env.reference_route()
    if frames is not None:
        frames.append(frame(env, -1, 0.0, False))
    total, latencies = 0.0, []
    while True:
        t0 = time.perf_counter()
        action = int(policy(obs))
        latencies.append((time.perf_counter() - t0) * 1000)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if frames is not None:
            frames.append(frame(env, action, reward, info["collision"]))
        if terminated or truncated:
            break
    return {
        "seed": seed, "success": bool(info["success"]), "end": info["end"],
        "scenario_id": info["scenario_id"], "difficulty": info["difficulty"],
        "beacons": [list(point) for point in env.beacons], "active_rocks": len(env.rocks),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "steps": env.steps, "visited": int(sum(info["visited"])),
        "battery": round(float(info["battery"]), 2), "collisions": int(info["collisions"]),
        "path_length": round(float(info["path_length"]), 3),
        "reference_route_m": round(reference_route_m, 3), "return": round(total, 4),
        "latency_ms_mean": round(float(np.mean(latencies)), 4),
        "latency_ms_max": round(float(np.max(latencies)), 4),
    }


def summarize(cases: list[dict], beacon_count: int) -> dict:
    ok = [c for c in cases if c["success"]]
    ref = reference_route(beacon_count)
    mean = lambda vals: round(float(np.mean(vals)), 4) if vals else None
    latencies = [c["latency_ms_mean"] for c in cases]
    difficulties = Counter(c.get("difficulty", "fixed") for c in cases)
    return {
        "episodes": len(cases),
        "success_rate": round(len(ok) / len(cases), 4),
        "failures_by_cause": {k: v for k, v in Counter(c["end"] for c in cases).items() if k != "exito"},
        "end_counts": {cause: sum(c["end"] == cause for c in cases) for cause in END_CAUSES},
        "route_efficiency": mean([
            c.get("reference_route_m", ref) / c["path_length"] for c in ok if c["path_length"] > 0
        ]),
        "reference_route_m": mean([c.get("reference_route_m", ref) for c in cases]),
        "episodes_by_difficulty": dict(sorted(difficulties.items())),
        "success_rate_by_difficulty": {
            difficulty: round(
                sum(c["success"] for c in cases if c.get("difficulty", "fixed") == difficulty)
                / count,
                4,
            )
            for difficulty, count in sorted(difficulties.items())
        },
        "collisions_per_episode": mean([c["collisions"] for c in cases]),
        "final_battery_success": mean([c["battery"] for c in ok]),
        "beacons_visited_mean": mean([c["visited"] for c in cases]),
        "steps_mean": mean([c["steps"] for c in cases]),
        "return_mean": mean([c["return"] for c in cases]),
        "return_std": round(float(np.std([c["return"] for c in cases])), 4),
        "latency_ms_mean": mean(latencies),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 4),
    }


def evaluate(policy, level_id: str, suite_id: str | None = None, *,
             episodes: int | None = None, policy_info: dict | None = None) -> dict:
    """Evalua la politica en el nivel con la suite fija y devuelve el resultado completo.

    episodes (opcional) limita la suite a sus primeras N semillas; el resultado
    lo marca como parcial y no sirve para comparar candidatos.
    """
    suite_id = suite_id or f"suite-{level_id}-v1"
    suite = load_suite(suite_id)
    if suite.get("level_id") not in (None, level_id):
        raise ValueError(f"la suite {suite_id} es del nivel {suite['level_id']}, no de {level_id}")
    seeds = suite["seeds"][:episodes] if episodes else suite["seeds"]
    level = load_level_config(level_id)
    environment = dict(suite.get("environment") or {})
    environment.setdefault("max_steps", int(suite.get("max_steps", 400)))
    env = BeaconExpeditionEnv(level=level_id, **environment)
    started = time.perf_counter()
    cases = [run_case(env, policy, seed) for seed in seeds]
    env.close()
    reserved = suite.get("seeds_source") == "reserved"
    return {
        "suite_id": suite_id, "suite_version": suite.get("version"), "suite_sha256": suite["suite_sha256"],
        "partial": len(seeds) < len(suite["seeds"]),
        "level_id": level_id, "environment_version": ENVIRONMENT_VERSION,
        "environment_hash": environment_hash(level_id, environment),
        "environment": environment,
        "git_commit": _git_commit(), "git_dirty": _git_dirty(),
        "policy": policy_info or (policy.describe() if hasattr(policy, "describe") else {"policy": str(policy)}),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(time.perf_counter() - started, 2),
        "metrics": summarize(cases, int(level["beacons"])),
        # El test reservado no publica sus semillas en los resultados.
        "cases": [{**c, "seed": "reservada"} for c in cases] if reserved else cases,
    }


def save_result(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.0%}" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def write_baselines_report(extra: list[tuple[str, object]] = ()) -> Path:
    """Evalua aleatorio y regla (y politicas extra) en N1, N2 y N3; escribe JSON y Markdown."""
    policies = [("random", RandomPolicy()), ("rule", RulePolicy()), *extra]
    rows = []
    for level_id in ("n1", "n2", "n3"):
        for name, policy in policies:
            result = evaluate(policy, level_id)
            out = save_result(result, RESULTS_DIR / f"baseline-{name}-{level_id}.json")
            rows.append((level_id, name, result, out))
            m = result["metrics"]
            print(f"{level_id} {name:10s} exito {m['success_rate']:.0%}  fallos {m['failures_by_cause']}")
    lines = [
        "# Resultados de los baselines",
        "",
        "Generado con `python pipelines/evaluate.py --baselines` (paso 2 del plan).",
        f"Commit `{rows[0][2]['git_commit'][:10]}` · ambiente `{ENVIRONMENT_VERSION}` · "
        "suites `suite-n1-v1`, `suite-n2-v1` y `suite-n3-v1` (50 episodios cada una, semillas fijas).",
        "",
        "La **tasa de éxito** mide la misión. El **retorno** (recompensa acumulada) se reporta",
        "aparte y solo se compara dentro de un mismo nivel: no equivale al éxito.",
        "",
        "| Nivel | Política | Éxito | Fallos por causa | Balizas | Pasos | Choques/ep. | Batería (éxitos) | Eficiencia | Retorno medio ± d.e. | Latencia media (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for level_id, name, result, out in rows:
        m = result["metrics"]
        fails = ", ".join(f"{k} {v}" for k, v in sorted(m["failures_by_cause"].items())) or "—"
        lines.append(
            f"| {level_id.upper()} | [{name}]({out.name}) | **{_fmt(m['success_rate'], True)}** | {fails} | "
            f"{_fmt(m['beacons_visited_mean'])} | {_fmt(m['steps_mean'])} | {_fmt(m['collisions_per_episode'])} | "
            f"{_fmt(m['final_battery_success'])} | {_fmt(m['route_efficiency'])} | "
            f"{m['return_mean']:.1f} ± {m['return_std']:.1f} | {_fmt(m['latency_ms_mean'])} |")
    lines += [
        "",
        "- **Balizas:** media de balizas visitadas en orden (N1 tiene 1, N2 y N3 tienen 2).",
        "- **Eficiencia:** ruta de referencia de centro a centro / distancia recorrida, solo en éxitos, "
        "en línea recta (N1: 17.5 m; N2 y N3: 24.5 m). Puede pasar de 1 porque basta con entrar al radio de 0.6 m.",
        "- **Latencia:** tiempo de decisión de la política por paso, sin contar la física.",
        "- Cada JSON enlazado tiene los resultados por caso (semilla, causa de fin, pasos, batería, "
        "choques, recorrido, retorno y latencia).",
    ]
    for name, policy in extra:
        info = policy.describe() if hasattr(policy, "describe") else {}
        if info.get("model_sha256"):
            lines.append(f"- **{name}:** modelo `{info.get('model_path')}`, SHA-256 `{info['model_sha256']}`.")
    report = RESULTS_DIR / "baselines.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evalua una politica con la suite fija de Robot Expedition")
    parser.add_argument("--policy", choices=["random", "rule", "qlearning", "ppo"], default="rule")
    parser.add_argument("--model", help="modelo: .npz (qlearning) o .zip (ppo)")
    parser.add_argument("--vecnormalize", help="vecnormalize.pkl del PPO (si se entreno con normalizacion)")
    parser.add_argument("--level", choices=["n1", "n2", "n3"], default="n2")
    parser.add_argument("--suite", help="id de suite (por defecto suite-<nivel>-v1; test-final = test reservado)")
    parser.add_argument("--episodes", type=int, help="limita la suite (resultado marcado como parcial)")
    parser.add_argument("--out", help="ruta del JSON de resultados")
    parser.add_argument("--baselines", action="store_true", help="aleatorio y regla en N1, N2 y N3 -> docs/resultados")
    parser.add_argument("--qlearning-model", help="con --baselines, agrega este Q-learning a la tabla")
    args = parser.parse_args(argv)

    if args.baselines:
        extra = [("qlearning", make_policy("qlearning", args.qlearning_model))] if args.qlearning_model else []
        print(f"Reporte: {write_baselines_report(extra)}")
        return

    policy = make_policy(args.policy, args.model, args.vecnormalize)
    result = evaluate(policy, args.level, args.suite, episodes=args.episodes)
    out = Path(args.out) if args.out else ROOT / "outputs" / "evaluaciones" / (
        f"{args.policy}-{args.level}-{result['suite_id']}-{int(time.time())}.json")
    save_result(result, out)
    m = result["metrics"]
    print(f"{args.policy} en {args.level.upper()} ({result['suite_id']}, {m['episodes']} episodios"
          f"{', PARCIAL' if result['partial'] else ''})")
    print(f"  tasa de exito        {m['success_rate']:.1%}")
    print(f"  fallos por causa     {m['failures_by_cause'] or '-'}")
    print(f"  balizas visitadas    {m['beacons_visited_mean']}")
    print(f"  eficiencia de ruta   {m['route_efficiency']}")
    print(f"  choques por episodio {m['collisions_per_episode']}")
    print(f"  bateria (exitos)     {m['final_battery_success']}")
    print(f"  retorno medio        {m['return_mean']} ± {m['return_std']}  (no es la tasa de exito)")
    print(f"  latencia media       {m['latency_ms_mean']} ms (p95 {m['latency_ms_p95']} ms)")
    print(f"Resultados por caso: {out}")


if __name__ == "__main__":
    main()
