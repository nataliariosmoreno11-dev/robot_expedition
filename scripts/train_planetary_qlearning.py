"""Entrena offline el Q-learning planetario por el mismo camino que usa la web.

Corre episodios completos contra la API en proceso (/sim/reset + /sim/step con
policy=planetary-qlearning), de modo que el checkpoint resultante es identico en
formato y recompensas al que produce produccion. Sirve para publicar un
checkpoint base que la API descarga al arrancar (PLANETARY_Q_SEED_URL).

Uso:
    python scripts/train_planetary_qlearning.py --out dist/planetary_qlearning.npz --rounds 6
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dist/planetary_qlearning.npz")
    parser.add_argument("--rounds", type=int, default=6, help="pasadas completas por los 8 niveles")
    parser.add_argument("--seeds-per-level", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="continua desde --out si ya existe")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        parser.error(f"{out} ya existe; usa --resume para continuar o elige otra ruta")
    work = Path(tempfile.mkdtemp(prefix="qlearning-train-"))
    # La API lee estas rutas al importarse: el entrenamiento nunca toca models/ ni logs/ del repo.
    os.environ["EXPEDITION_DATA_DIR"] = str(work)
    os.environ["PLANETARY_Q_CHECKPOINT"] = str(out)
    os.environ["PLANETARY_Q_SEED_URL"] = ""
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    started = time.time()
    seed = 1000
    for round_number in range(1, args.rounds + 1):
        for level in range(1, 9):
            wins = 0
            for _ in range(args.seeds_per_level):
                seed += 1
                reset = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": level,
                                                       "seed": seed, "policy": "planetary-qlearning"})
                reset.raise_for_status()
                sid = reset.json()["session_id"]
                while True:
                    step = client.post("/sim/step", json={"session_id": sid})
                    step.raise_for_status()
                    data = step.json()
                    if data["terminated"] or data["truncated"]:
                        wins += bool(data["success"])
                        break
            status = client.get("/training/qlearning").json()
            print(f"ronda {round_number} · nivel {level} · exito {wins}/{args.seeds_per_level} · "
                  f"ep {status['episodes']} · estados {status['states']} · eps {status['epsilon']:.3f} · "
                  f"{time.time() - started:.0f}s", flush=True)
    print(json.dumps(client.get("/training/qlearning").json(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
