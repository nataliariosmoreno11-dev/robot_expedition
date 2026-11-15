"""Compare physics throughput with identical seeds/actions and full step recording."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv


def benchmark():
    results = {}
    traces = {}
    with tempfile.TemporaryDirectory() as directory:
        for backend in ('dummy', 'subproc'):
            def factory():
                return PlanetaryExplorationEnv(history_steps=4, difficulty_level=3,
                    record_episodes=True, episode_db=Path(directory) / f'{backend}.sqlite',
                    campaign=True, curriculum=False, training_no_progress_limit=80)
            # Precreate schema to exclude schema setup contention from timings.
            factory().close()
            before = time.perf_counter()
            env = (SubprocVecEnv([factory] * 4, start_method='spawn') if backend == 'subproc'
                   else DummyVecEnv([factory] * 4))
            env.seed(42)
            env.reset()
            startup = time.perf_counter() - before
            rng = np.random.default_rng(13)
            trace = []
            before = time.perf_counter()
            for _ in range(512):
                obs, reward, done, _ = env.step(rng.integers(0, 7, size=4))
                # DummyVecEnv casts rewards to float32; SubprocVecEnv preserves float64.
                trace.append(np.concatenate([obs.flatten(), reward, done]).astype(np.float32))
            elapsed = time.perf_counter() - before
            env.close()
            traces[backend] = np.array(trace)
            results[backend] = {'steps': 2048, 'seconds': elapsed,
                                'steps_per_second': 2048 / elapsed, 'startup_seconds': startup}
            print(backend, results[backend], flush=True)
    results['identical_transitions'] = bool(np.array_equal(traces['dummy'], traces['subproc']))
    results['speedup'] = results['dummy']['seconds'] / results['subproc']['seconds']
    Path('docs/evidencia/vector-throughput.json').write_text(json.dumps(results, indent=2))
    assert results['identical_transitions'], 'Parallel stepping changed simulation results'


if __name__ == '__main__':
    benchmark()
