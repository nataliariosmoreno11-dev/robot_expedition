# Almacenamiento compacto del entrenamiento

Respaldo publicado:
https://github.com/nataliariosmoreno11-dev/robot_expedition/releases/tag/training-compact-20261115

- `training-weights-20261115.zip`: 19 versiones verificadas, hasta 3.345.408 pasos,
  con pesos, normalización, optimizador, configuración, currículo, evaluaciones,
  código registrado y puntos de retención. No es el modelo final de 10 escenarios.
- `training-history-20261115.zip`: base SQLite con 24.826 métricas y 602 replays
  seleccionados. Las métricas originales se compararon por hash antes de reemplazar
  la base. Los pasos de intentos repetitivos no forman parte de este respaldo.
- `SHA256SUMS.json`: hashes de los dos archivos ZIP.

Se conservan localmente los modelos, la base compacta y los replays del visor.
Se eliminaron snapshots repetidos de SQLite, checkpoints intermedios redundantes
y la caché DVC anterior. Los punteros DVC se regeneraron para los modelos compactos;
los antiguos snapshots completos de pasos no se pueden recuperar desde la caché
local. Los pesos respaldados permanecen disponibles en GitHub Releases.

Para limitar el crecimiento futuro, los episodios de campaña conservan dos
intentos por versión, dificultad, modo y resultado, además de los 50 más recientes
y los protegidos manualmente. Todas las métricas escalares se conservan. SQLite
reutiliza el espacio de los replays descartados. Cada versión apunta a la base
compartida mediante `episode-store.json`; ya no copia todo el historial.

Para descargar el respaldo:

```sh
gh release download training-compact-20261115 \
  --repo nataliariosmoreno11-dev/robot_expedition --dir respaldo
```

Verifica los hashes antes de extraer. Los ZIP contienen rutas relativas a la raíz
del repositorio. Detén entrenador y API antes de restaurar la base de episodios.
Para consultar o reanudar el entrenamiento, consulta `docs/training-terminal.md`.

En WSL, borrar archivos libera espacio dentro de Linux. Para devolverlo a Windows
también puede ser necesario compactar el VHDX, con Ubuntu detenido. Esta instalación
soporta `wsl --manage Ubuntu --compact`; no debe ejecutarse mientras se entrena.
