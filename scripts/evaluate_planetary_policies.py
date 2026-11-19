"""Evalúa el guía planificador y el Q-learning planetario por el camino de la API.

Corre episodios completos con /sim/reset + /sim/step en los 8 niveles, con el
límite de pasos de cada escenario. El Q-learning se evalúa sobre una COPIA del
checkpoint indicado: este script nunca modifica checkpoints ni el historial.

Uso:
    python scripts/evaluate_planetary_policies.py --episodes 10
    python scripts/evaluate_planetary_policies.py --q-checkpoint planetary_qlearning.npz --output resultado.json
"""
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=10, help="episodios por nivel y política")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--seed-start", type=int, default=6000, help="distinto de las semillas de entrenamiento")
    parser.add_argument("--policies", default="planetary-rule,planetary-qlearning")
    parser.add_argument("--q-checkpoint", help="checkpoint Q a evaluar (se copia; por defecto, el base publicado)")
    parser.add_argument("--output", type=Path, help="JSON opcional con el resumen")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="planetary-eval-"))
    (work / "models").mkdir()
    if args.q_checkpoint:
        shutil.copy(args.q_checkpoint, work / "models" / "planetary_qlearning.npz")
        os.environ["PLANETARY_Q_SEED_URL"] = ""
    os.environ["EXPEDITION_DATA_DIR"] = str(work)
    os.environ.pop("PLANETARY_Q_CHECKPOINT", None)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import api.main as api_main

    client = TestClient(api_main.app)
    levels = [int(item) for item in args.levels.split(",")]
    summary: dict = {"episodes_per_level": args.episodes, "seed_start": args.seed_start,
                     "q_seed": api_main.PLANETARY_Q_SEED_STATUS, "policies": {}}
    for policy in args.policies.split(","):
        rows = {}
        for level in levels:
            ends, successes, steps, battery, collisions = collections.Counter(), 0, [], [], []
            for k in range(args.episodes):
                reset = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": level,
                                                       "seed": args.seed_start + k, "policy": policy})
                reset.raise_for_status()
                sid = reset.json()["session_id"]
                while True:
                    data = client.post("/sim/step", json={"session_id": sid}).json()
                    if data["terminated"] or data["truncated"]:
                        break
                with api_main.sessions_lock:
                    session = api_main.sessions.pop(sid, None)
                if session is not None:
                    collisions.append(session.env.collisions)
                    session.env.close()
                ends[data["end"]] += 1
                successes += bool(data["success"])
                steps.append(data["steps"])
                battery.append(data["battery"])
            rows[f"L{level}"] = {"success": successes, "episodes": args.episodes, "ends": dict(ends),
                                 "mean_steps": round(sum(steps) / len(steps), 1),
                                 "mean_battery_left": round(sum(battery) / len(battery), 1),
                                 "mean_collisions": round(sum(collisions) / max(1, len(collisions)), 2)}
            row = rows[f"L{level}"]
            print(f"{policy:<20} L{level}: {successes}/{args.episodes} éxito · {row['mean_steps']} pasos · "
                  f"batería {row['mean_battery_left']} % · choques {row['mean_collisions']} · {dict(ends)}",
                  flush=True)
        total = sum(row["success"] for row in rows.values())
        summary["policies"][policy] = {"levels": rows, "success": total, "episodes": args.episodes * len(levels)}
        print(f"{policy:<20} TOTAL {total}/{args.episodes * len(levels)}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
