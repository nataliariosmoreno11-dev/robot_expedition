"""Package completed local checkpoints without logs, credentials or episode databases."""
import hashlib
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['8a5ed5c76b1b40f9bbeea0ad736fb2bb', '2ae171ad14ef45c699e1cb411da59fd2']
FILES = ['ppo_model.zip', 'vecnormalize.pkl', 'optimizer.pt', 'checkpoint_state.json',
         'curriculum_state.json', 'config.json', 'sha256.json', 'evaluacion.json',
         'resume-selection.json', 'retention_guard.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=RUNS)
    args = parser.parse_args()
    output = ROOT / 'logs/publication'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'robot-expedition-planetary-checkpoints.zip'
    sums = {}
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for run in args.runs:
            if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
                raise ValueError('Invalid run identifier')
            directory = ROOT / 'models' / run
            declared = json.loads((directory / 'sha256.json').read_text())
            for name, expected in declared.items():
                actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Checkpoint integrity failure: {run}/{name}')
            paths = [directory / name for name in FILES]
            paths.append(ROOT / 'manifests' / f'{run}.json')
            for path in paths:
                if not path.exists():
                    continue
                relative = path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                sums[relative] = hashlib.sha256(data).hexdigest()
                bundle.writestr(relative, data)
        bundle.writestr('SHA256SUMS.txt', ''.join(f'{digest}  {name}\n' for name, digest in sums.items()))
        bundle.writestr('README.txt',
            'PRELIMINARY PLANETARY PPO CHECKPOINTS — NOT THE FINAL TEN-SCENARIO MODEL.\n'
            'Observation: 132 values (4 x 33); actions: 7. Use the planetary training API.\n'
            'This archive is not compatible with the historical beacon /evaluate endpoint.\n'
            'Extract into the repository root; verify SHA256SUMS.txt before loading.\n'
            'For inference load ppo_model.zip AND vecnormalize.pkl together.\n'
            'Optimizer and checkpoint state support resuming training.\n'
            'See evaluacion.json for actual performance; mastered curriculum is not a final certificate.\n'
            'Episode databases and intermediate checkpoints remain on the training machine.\n')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n')
    (output / 'release-notes.md').write_text(
        'Checkpoints preliminares del agente PPO planetario. **No es el modelo final de los diez escenarios.**\n\n'
        + ''.join(f'- `{run}`: pesos y resultados incluidos.\n' for run in args.runs) + '\n'
        'El ZIP incluye pesos, normalización, optimizador, configuración, currículo, manifiestos y evaluaciones. '
        'SHA-256 del ZIP y de los archivos incluidos para verificar la descarga. Las bases SQLite de pasos y los checkpoints '
        'intermedios permanecen locales.\n\n'
        'Contrato planetario: 132 observaciones y 7 acciones. No cargar en el endpoint histórico de balizas `/evaluate`. '
        'Publicar los archivos no migra el entrenamiento ni actualiza automáticamente el servidor desplegado.\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest}))


if __name__ == '__main__':
    main()
