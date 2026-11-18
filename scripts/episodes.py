"""Consulta y exporta episodios registrados sin volver a ejecutar la política.

Ejemplos:
  python scripts/episodes.py list --limit 20
  python scripts/episodes.py show 3427
  python scripts/episodes.py replay 3427 --output replay-3427.json
  python scripts/episodes.py scenario 3427
  python scripts/episodes.py compare 100 500 1000
  python scripts/episodes.py status
  python scripts/episodes.py metrics
  python scripts/episodes.py protect 3427
  python scripts/episodes.py unprotect 3427
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_expedition.planetary.episode_store import DEFAULT_DB, EpisodeStore  # noqa: E402


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Explorador de episodios planetarios")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--run-id")

    for name in ("show", "scenario"):
        command = commands.add_parser(name)
        command.add_argument("episode", type=int)

    replay = commands.add_parser("replay")
    replay.add_argument("episode", type=int)
    replay.add_argument("--output", type=Path)

    compare = commands.add_parser("compare")
    compare.add_argument("episodes", type=int, nargs="+")
    commands.add_parser("status")
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--run-id")
    for name in ("protect", "unprotect"):
        command = commands.add_parser(name)
        command.add_argument("episode", type=int)

    history=commands.add_parser("history")
    history.add_argument("scenario_id")
    history.add_argument("--offset",type=int,default=0)
    history.add_argument("--limit",type=int,default=100)
    report=commands.add_parser("progress")
    report.add_argument("scenario_id")
    report.add_argument("--mode",default="entrenamiento")
    attempts=commands.add_parser("attempts")
    attempts.add_argument("scenario_id")
    attempts.add_argument("attempts",type=int,nargs="+")
    args = parser.parse_args()
    store = EpisodeStore(args.db)
    try:
        if args.command == "history":
            result=store.get_scenario_history(args.scenario_id,limit=args.limit,offset=args.offset)
        elif args.command == "progress":
            result=store.scenario_report(args.scenario_id,mode=args.mode)
        elif args.command == "attempts":
            result=store.compare_scenario_attempts(args.scenario_id,*args.attempts)
        elif args.command == "list":
            result = store.list_episodes(limit=args.limit, offset=args.offset, run_id=args.run_id)
        elif args.command == "show":
            result = store.get_episode(args.episode)
        elif args.command == "scenario":
            result = store.load_scenario(args.episode)
        elif args.command == "replay":
            result = store.replay_episode(args.episode)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                print(args.output)
                return
        elif args.command == "status":
            result = store.storage_status()
        elif args.command == "metrics":
            result = store.metrics(run_id=args.run_id)
        elif args.command in ("protect", "unprotect"):
            result = store.protect_episode(args.episode, args.command == "protect")
        else:
            result = store.compare_episodes(*args.episodes)
        _print(result)
    finally:
        store.close()


if __name__ == "__main__":
    main()
