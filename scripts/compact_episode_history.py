"""Build a small replay database, retaining all metrics and representative attempts.

Does not replace or delete the source database. Stop writers before using the
result to replace the live database.
"""
import argparse
from pathlib import Path
import sqlite3


def compact(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination:
        raise ValueError("Destination must be a new file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as original:
        schema = original.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    with sqlite3.connect(destination) as target:
        for (sql,) in schema:
            target.execute(sql)
        target.execute("ATTACH DATABASE ? AS original", (f"file:{source}?mode=ro",))
        target.execute("CREATE TEMP TABLE keep_episode(number INTEGER PRIMARY KEY)")
        target.execute("""INSERT OR IGNORE INTO keep_episode
            SELECT episode_number FROM (
                SELECT episode_number, ROW_NUMBER() OVER (
                    PARTITION BY run_id, difficulty_level, mode, success ORDER BY episode_number DESC
                ) AS rank FROM original.episodes WHERE ended_at IS NOT NULL
            ) WHERE rank <= 2""")
        target.execute("INSERT OR IGNORE INTO keep_episode SELECT episode_number FROM original.episodes WHERE protected=1")
        target.execute("INSERT OR IGNORE INTO keep_episode SELECT episode_number FROM original.episodes ORDER BY episode_number DESC LIMIT 50")
        target.execute("INSERT INTO metadata SELECT * FROM original.metadata")
        target.execute("INSERT INTO training_metrics SELECT * FROM original.training_metrics")
        target.execute("INSERT INTO episodes SELECT * FROM original.episodes WHERE episode_number IN (SELECT number FROM keep_episode)")
        target.execute("INSERT INTO episode_steps SELECT * FROM original.episode_steps WHERE episode_number IN (SELECT number FROM keep_episode)")
        target.execute("UPDATE training_metrics SET replay_retained = (episode_number IN (SELECT number FROM keep_episode))")
        indexes = target.execute("SELECT sql FROM original.sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchall()
        for (sql,) in indexes:
            target.execute(sql)
        # Prevent reusing episode identifiers that have been removed.
        maximum = target.execute("SELECT MAX(episode_number) FROM original.episodes").fetchone()[0] or 0
        target.execute("UPDATE sqlite_sequence SET seq=? WHERE name='episodes'", (maximum,))
        target.commit()
        target.execute("DETACH DATABASE original")
        assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not target.execute("PRAGMA foreign_key_check").fetchall()
        return {"episodes": target.execute("SELECT count(*) FROM episodes").fetchone()[0],
                "metrics": target.execute("SELECT count(*) FROM training_metrics").fetchone()[0],
                "bytes": destination.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(compact(args.source, args.destination))
