"""Detached SMTP notifications for first-time curriculum promotions (WSL).

Configuration and delivery state live under ignored logs/training_notifications.
No credentials are stored in source control. Restart is safe; pending messages retry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import smtplib
import ssl
import time
from email.message import EmailMessage

ROOT = Path(__file__).resolve().parents[1]


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def promotions(root):
    found = {}
    for path in sorted((root / 'models').glob('*/mastered-scenario-*/curriculum_state.json')):
        state = read(path, {})
        level = int(path.parent.name.split('-')[2])
        if level in state.get('mastered', []):
            found.setdefault(str(level), {
                'level': level, 'checkpoint': str(path.parent.relative_to(root)),
                'evaluation': state.get('evaluations', {}).get(str(level), {}),
            })
    return found


def collect(state, found, recipients):
    for level, event in found.items():
        if level not in state['seen']:
            state['events'][level] = {**event, 'pending': list(recipients), 'sent': []}
            state['seen'].append(level)


def deliver(config, event, recipient):
    message = EmailMessage()
    message['From'] = config['sender']
    message['To'] = recipient
    message['Subject'] = f"Robot Expedition: escenario {event['level']} aprobado"
    evidence = event['evaluation']
    message.set_content(
        f"El currículo aprobó el escenario {event['level']}.\n"
        f"Pasos: {evidence.get('training_step', 'sin dato')}\n"
        f"Éxito: {evidence.get('success_rate', 'sin dato')}\n"
        f"Episodios evaluados: {evidence.get('episodes', 'sin dato')}\n"
        f"Checkpoint local: {event['checkpoint']}\n\n"
        "Esta aprobación no equivale a la certificación final de los 10 escenarios.\n"
        "El entrenamiento puede reforzar escenarios anteriores si detecta regresiones.\n"
    )
    context = ssl.create_default_context()
    if config.get('tls', 'starttls') == 'ssl':
        client = smtplib.SMTP_SSL(config['host'], config.get('port', 465), timeout=20, context=context)
    else:
        client = smtplib.SMTP(config['host'], config.get('port', 587), timeout=20)
    with client:
        if config.get('tls', 'starttls') != 'ssl':
            client.starttls(context=context)
        client.login(config['username'], config['password'])
        client.send_message(message)


def tick(root=ROOT):
    directory = root / 'logs/training_notifications'
    config = read(directory / 'config.json', {})
    state = read(directory / 'state.json', None)
    found = promotions(root)
    if state is None:
        # Existing approvals establish the baseline; no retrospective flood.
        state = {'seen': list(found), 'events': {}}
    collect(state, found, config.get('recipients', []))
    state.update(pid=os.getpid(), updated_at=time.time(), status='waiting_for_smtp')
    ready = all(config.get(k) for k in ('host', 'sender', 'username', 'password', 'recipients'))
    if ready:
        state['status'] = 'monitoring'
    write(directory / 'state.json', state)
    if ready:
        for event in state['events'].values():
            for recipient in list(event['pending']):
                try:
                    deliver(config, event, recipient)
                except Exception as exc:
                    # Do not log SMTP responses: they can contain addresses/auth details.
                    state.update(status='delivery_error', error_type=type(exc).__name__)
                else:
                    event['pending'].remove(recipient)
                    event['sent'].append(recipient)
                write(directory / 'state.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'logs/training_notifications'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            tick()
            if args.once:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
