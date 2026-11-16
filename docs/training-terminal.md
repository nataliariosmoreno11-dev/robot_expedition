# Entrenamiento desde la terminal de Ubuntu / WSL

No requiere abrir la página ni arrancar la API. Desde Ubuntu:

```sh
cd /home/tom/code/robot_expedition
.venv/bin/python scripts/training_control.py status
.venv/bin/python scripts/training_control.py resume
.venv/bin/python scripts/training_control.py watch
```

- `status`: estado real de los procesos, pasos, escenarios y antigüedad del progreso.
- `resume`: verifica los hashes de pesos, normalizador, optimizador y contador;
  continúa desde el checkpoint completo más reciente sin duplicar procesos.
  Si el último guardado está incompleto, usa una versión anterior verificable.
  No inicia una política desde cero. Si el entrenador sigue vivo pero perdió su
  supervisor, recupera la supervisión sin modificar sus pesos.
- `watch`: actualiza cada 5 segundos. Ctrl+C cierra solamente el monitor.
- `logs`: muestra el final de los registros del supervisor y del entrenador.
- `stop`: solicita detenerse y guardar. Comprueba con `status` que los procesos
  hayan terminado antes de apagar Ubuntu.

Para detener y guardar:

```sh
.venv/bin/python scripts/training_control.py stop
.venv/bin/python scripts/training_control.py watch
```

El contador puede permanecer estable durante evaluaciones, comprobaciones de
retención y guardado. Un proceso vivo no garantiza que aprenda: consulta también
los escenarios aprobados y los resultados de evaluación en los artefactos.

El entrenamiento se ejecuta separado de esta terminal. Cerrar el monitor o la
terminal no solicita una parada, pero suspender/apagar el PC, cerrar WSL o
reiniciar Ubuntu sí puede interrumpirlo. Después vuelve a ejecutar `resume`.

Desde PowerShell también puedes ejecutar, por ejemplo:

```powershell
wsl -d Ubuntu --cd /home/tom/code/robot_expedition -- .venv/bin/python scripts/training_control.py resume
wsl -d Ubuntu --cd /home/tom/code/robot_expedition -- .venv/bin/python scripts/training_control.py watch
```

Los checkpoints están bajo `models/<versión>/`. Los registros están en
`logs/training_jobs/<ejecución>/worker.log` y
`logs/autonomous_campaign/supervisor.log`.
