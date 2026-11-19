# Versiones aprobadas: escenarios 1 a 8

La campaña activa contiene ocho mapas. Se retiraron los escenarios 9 y 10.
Los mapas 1 a 8, las observaciones y las reglas de batería no cambiaron.
La escala de dificultad de las observaciones sigue siendo nivel/10 y la red
conserva sus dimensiones para cargar los pesos existentes sin reinterpretarlos.
Las dos ramas internas sobrantes de las redes antiguas no se utilizan.

El catálogo `docs/approved-models.json` identifica una versión por avance:

| Versión | Escenarios aprobados |
|---|---|
| f7c104a304 | 1 |
| d0ecdae668 | 1, 2 |
| d0c438b791 | 1, 2, 3 |
| bdbc0d04ff | 1, 2, 3, 4 |
| 646fbc7070 | 1, 2, 3, 4, 5 |
| 156aaf2054 | 1, 2, 3, 4, 5, 6 |
| 6109346a10 | 1, 2, 3, 4, 5, 6, 7 |
| 4bcfd8f641 | 1, 2, 3, 4, 5, 6, 7, 8 |

Son copias de los checkpoints exactos aprobados, no los pesos posteriores de
cada ejecución. Cada aprobación exige al menos 50 misiones completas, éxito
mínimo del 85%, media de choques <=2 y retención de escenarios anteriores.
El origen, pasos y SHA-256 se incluyen en cada `approved-version.json`.
Las evaluaciones originales disponibles se conservan en `evidence/`.

Los ocho paquetes pequeños se incluyen directamente en Git, con pesos PPO,
normalizador, optimizador, configuración y comprobaciones SHA-256. No necesitan
recuperar estos pesos desde un remoto DVC. La API reconstruye su registro local
al consultar las versiones después de clonar. Para ejecutar las redes por
escenario utiliza `POLICY_BACKEND=torch`.

La página muestra únicamente estas versiones en los selectores. El historial
de recorridos consulta la ejecución de origen y necesita su base SQLite local;
los pesos sí permiten generar evaluaciones nuevas sin ese historial.

`python -m scripts.curate_models` exporta los hitos y prepara el informe de
limpieza. `--prune` elimina los pesos intermedios, conservando informes,
versiones aprobadas, el modelo auditado y los puntos de reanudación de la campaña.
El informe de archivos eliminados queda en `logs/model-cleanup.json`.
