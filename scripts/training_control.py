"""Control local de la campaña desde Ubuntu/WSL, sin API ni navegador."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"starting", "running", "stopping"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def alive(state, module):
    """Check the command, not just a PID that WSL may have reused."""
    try:
        pid = int(state.get("pid", 0))
        if pid <= 0:
            return False
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if module.encode() not in command:
            return False
        if module == "api.training_worker":
            return str(ROOT / "logs/training_jobs" / state["job_id"]).encode() in command
        return True
    except (OSError, ValueError, KeyError):
        return False


def jobs():
    return [(path, read(path)) for path in (ROOT / "logs/training_jobs").glob("*/state.json")]


def verified_checkpoint(job):
    run_id = job.get("run_id", "")
    if len(run_id or "") != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("sin versión de pesos")
    model = ROOT / "models" / run_id
    hashes = read(model / "sha256.json")
    for name in ("ppo_model.zip", "vecnormalize.pkl", "optimizer.pt", "checkpoint_state.json"):
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != hashes.get(name):
            raise ValueError(f"checkpoint incompleto o alterado: {name}")
    config = read(model / "config.json").get("environment", {})
    if config.get("history_steps") != 4 or not config.get("campaign"):
        raise ValueError("checkpoint incompatible")
    return int(read(model / "checkpoint_state.json")["total_timesteps"])


def snapshot():
    campaign = read(ROOT / "logs/autonomous_campaign/state.json")
    job_id = campaign.get("active_job_id") or campaign.get("parent_job_id")
    job = read(ROOT / "logs/training_jobs" / str(job_id) / "state.json")
    if not alive(job, "api.training_worker"):
        live_jobs = [value for _, value in jobs() if alive(value, "api.training_worker")]
        if len(live_jobs) == 1:
            job = live_jobs[0]
    model = ROOT / "models" / str(job.get("run_id"))
    progress = read(model / "progress.json") or campaign.get("progress", {})
    curriculum = read(model / "curriculum_state.json") or campaign
    supervisor = alive(campaign, "api.campaign_worker")
    worker = alive(job, "api.training_worker")
    status = "ACTIVO" if worker else "SUPERVISOR ACTIVO" if supervisor else "DETENIDO"
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} · {status}",
             f"Pasos registrados: {progress.get('total_timesteps', 0):,}",
             f"Escenarios aprobados en currículo: {curriculum.get('mastered', [])}",
             f"Escenario actual: {curriculum.get('current_level', '?')} · Estado guardado: {job.get('status', 'sin ejecución')}",
             f"Versión: {job.get('run_id') or 'preparando'}",
             f"Supervisor: {'activo' if supervisor else 'detenido'} · Entrenador: {'activo' if worker else 'detenido'}"]
    path = model / "progress.json"
    if path.exists():
        lines.append(f"Última actualización de pasos: hace {int(time.time() - path.stat().st_mtime)} s (puede pausarse durante evaluaciones/guardado)")
    reports = list(model.glob("periodic-evaluation-*.json"))
    if reports:
        latest = max(reports, key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        report = read(latest)
        metrics = report.get("metrics", {})
        lines.append(f"Evaluación nivel {report.get('scenario_level', '?')} a {report.get('training_step', 0):,} pasos: "
                     f"{metrics.get('success_rate', 0):.0%} éxitos en {metrics.get('episodes', 0)} intentos · "
                     f"{metrics.get('samples_collected_mean', 0):.2f} muestras · "
                     f"{metrics.get('collisions_per_episode', 0):.2f} choques de media")
        if report.get("early_rejection"):
            lines.append("Suite rechazada anticipadamente: ya no podía alcanzar el mínimo de aprobación.")
    else:
        lines.append("Esta versión aún no tiene una evaluación periódica; los pasos no prueban aprobación.")
    guard = read(model / "retention_guard.json")
    if guard:
        lines.append(f"Retención: {'aceptada' if guard.get('passed') else 'rechazada; pesos seguros restaurados'} · "
                     f"Último punto seguro: {guard.get('safe_step', 0):,} pasos")
    if not supervisor and not worker:
        lines.append("Para reanudar: .venv/bin/python scripts/training_control.py resume")
    if job.get("error"):
        lines.append(f"Último error: {job['error']}")
    return "\n".join(lines)


def resume():
    directory = ROOT / "logs/autonomous_campaign"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(directory / "state.json")
        if alive(state, "api.campaign_worker"):
            print("La campaña ya está activa. No se inicia otra copia.")
            return
        if state.get("status") == "completed":
            print("La campaña ya terminó. Consulta models/final-release.json.")
            return
        # Respect a supervisor still acquiring/publishing its PID.
        with (directory / "supervisor.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            entries = jobs()
            live = [job for _, job in entries if alive(job, "api.training_worker")]
            if len(live) > 1:
                raise ValueError("Hay varios entrenadores activos; no se iniciará otro.")
            if live and (ROOT / 'logs/training_jobs' / live[0]['job_id'] / 'stop').exists():
                raise ValueError("El entrenador está guardando para detenerse. Espera y ejecuta resume de nuevo.")
            if live:
                parent = live[0].get("parent_job_id") or live[0]["job_id"]
                state.update(active_job_id=live[0]["job_id"], parent_job_id=parent)
                print("Recuperando supervisión del entrenador activo; sus pesos no se modifican.")
            else:
                candidates = []
                for _, job in entries:
                    try:
                        steps = verified_checkpoint(job)
                        model = ROOT / "models" / job["run_id"]
                        mastered = len(read(model / "curriculum_state.json").get("mastered", []))
                        candidates.append((steps, mastered, (model / "checkpoint_state.json").stat().st_mtime_ns, job["job_id"]))
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                if not candidates:
                    raise ValueError("No hay checkpoint completo y verificable. No se iniciará desde cero.")
                steps, _, _, parent = max(candidates)
                print(f"Checkpoint verificado: {steps:,} pasos · ejecución {parent}")
                print("Se usa la versión completa más reciente; cualquier guardado incompleto se conserva y se omite.")
                state.setdefault("rounds", []).append({"job_id": state.get("active_job_id"), "status": "interrupted", "reason": "terminal recovery"})
                state.update(active_job_id=None, parent_job_id=parent)
            write(directory / f"before-resume-{time.time_ns()}.json", read(directory / "state.json"))
            for path, job in entries:
                if job.get("status") in ACTIVE and not alive(job, "api.training_worker"):
                    job.update(status="interrupted", error="Proceso detenido; recuperación desde terminal")
                    write(path, job)
            state.update(status="starting")
            state.pop("error", None)
            write(directory / "state.json", state)
            # 'resume' explicitly authorizes continuing an intentionally stopped campaign.
            (directory / "stop").unlink(missing_ok=True)
        with (directory / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "api.campaign_worker", "--parent-job", parent,
                 "--budget", str(state.get("budget_per_run", 262144)), "--resume"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
        time.sleep(1)
        if process.poll() is not None:
            raise ValueError("El supervisor terminó al iniciar. Consulta el comando logs.")
        print(f"Supervisor iniciado en segundo plano (PID {process.pid}). Comprueba watch para confirmar avances.")


def stop():
    directory = ROOT / "logs/autonomous_campaign"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stop").touch()
    for path, job in jobs():
        if alive(job, "api.training_worker"):
            (path.parent / "stop").touch()
    print("Parada solicitada. Espera a que termine de guardar antes de apagar WSL.")


def logs():
    directory = ROOT / "logs/autonomous_campaign"
    state = read(directory / "state.json")
    for path in [directory / "supervisor.log", ROOT / "logs/training_jobs" / str(state.get("active_job_id")) / "worker.log"]:
        print(f"\n{path.relative_to(ROOT)}")
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 6000))
                print(stream.read().decode("utf-8", errors="replace"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "watch", "resume", "stop", "logs"])
    args = parser.parse_args()
    try:
        if args.command == "watch":
            while True:
                if sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print(snapshot(), flush=True)
                print("\nCtrl+C cierra este monitor; el entrenamiento continúa.", flush=True)
                time.sleep(5)
        elif args.command == "status":
            print(snapshot())
        else:
            {"resume": resume, "stop": stop, "logs": logs}[args.command]()
    except KeyboardInterrupt:
        print("\nMonitor cerrado. No se solicitó detener el entrenamiento.")
    except (OSError, ValueError, KeyError) as exc:
        print(f"No se pudo completar la operación: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
