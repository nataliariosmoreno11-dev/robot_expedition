# Contratos del proyecto

Este documento fija la interfaz común descrita en la sección 3 de
`docs/plan-proyecto-final.md`. Los cambios incompatibles requieren actualizar
este documento y avisar al equipo.

## Niveles y configuración

N1 usa una baliza (B1), dos rocas y terreno sin dunas; el agente visita B1 y
regresa a la base. N2 usa B1, B2, cinco rocas y dunas. N3 es N2 con bloqueo del
corredor al visitar B1, ruido de rayos σ=0.08 m y el rayo central muerto.

Cada YAML contiene `level_id`, `beacons`, `active_rocks`, `dunes`, `blocking`,
`ray_noise` y `dead_ray`. Todos los niveles conservan observación de 20
`float32` y `Discrete(5)`. La versión del ambiente es
`robot-expedition-3d/1.1.0`; el hash SHA-256 de código, nivel y opciones se
obtiene con `environment_hash(config, environment)`.

Un experimento puede declarar `environment.scenario_set: varied-v1`. En ese
caso cada reset selecciona un escenario reproducible con balizas, rocas y
dificultad propias. `map_spec()` añade `scenario_id` y `difficulty`; no cambia
los campos existentes ni el contrato `Box(20)`/`Discrete(5)`.

**Etapa** (one-hot de la observación, índices 15–17, y campo `etapa` de los
cuadros): 1 = ir a B1, 2 = ir a B2, 3 = volver a la base. La etapa 3 significa
siempre "volver a la base", también en N1, donde se pasa directo de 1 a 3. Así
un modelo entiende la observación igual en los tres niveles.

## Configuración de experimento

Los experimentos viven en `configs/experiments/exp-XXX.yaml` y declaran
hipótesis, nivel, algoritmo, presupuesto, semillas, hiperparámetros,
normalización y `parent_run_id`.

## Manifiesto

Cada `manifests/<run_id>.json` debe incluir `run_id`, `experiment_id`,
`level_id`, `algorithm`, `environment_version`, `git_commit`,
`config_sha256`, `model_uri`, `model_sha256`, `normalization_uri`,
`obs_spec`, `action_spec`, `runtime_ref`, `evaluation_suite_id`, `seed`,
`budget_timesteps`, `parent_run_id`, `status` y `metrics`. Los estados válidos
son `completed`, `failed`, `interrupted` y `rejected`.

## Catálogo

`catalog/catalog.json` lo publica `pipelines/catalog.py` (o el flow de
Prefect, `pipelines/flow.py`) a partir de los manifiestos y de las
evaluaciones por nivel.

| Campo | Contenido |
|---|---|
| `catalog_version`, `served_version` | Entero creciente y su etiqueta (`catalog-vN`). Publicar el mismo contenido no crea versión nueva |
| `content_sha256`, `previous_version` | Huella del contenido y versión anterior |
| `published_at`, `published_by`, `git_commit` | Quién publicó, cuándo y desde qué commit |
| `selection_criteria` | Ruta, versión y SHA-256 de `configs/selection.yaml` |
| `recommended_run_id` | `{n1, n2, n3}`: el run recomendado por nivel (o `null`) |
| `pins` | Niveles fijados por un rollback: `{run_id, from_run_id, reason, at}` |
| `decisions` | Por nivel, la decisión y el motivo de cada candidato (recomendado, rechazado, no elegible, se mantiene, rollback) |
| `runs[]` | Campos del manifiesto + `hypothesis`, `change`, `reference`, `training_time_s`, `commit_url`, `mlflow_run_id`, `metrics.success_rate` (en su nivel de entrenamiento), `metrics_by_level` (métricas de la suite de **cada** nivel con `results_path`) y `curves` (curvas de entrenamiento resumidas) |

- **Resultados por caso:** `catalog/results/<run_id>/<nivel>.json`. Solo se
  reutilizan si coinciden el SHA-256 del modelo, el de la suite y el hash del
  ambiente.
- **Historial:** `catalog/history.jsonl`, una línea por publicación, rollback
  o quitada de pin (`actor`, `at`, `recommended_before`, `recommended_after` y
  `reason`).

## API

| Ruta | Contrato |
|---|---|
| `GET /health` | Estado del servicio |
| `GET /version` | Commit, release, versión del ambiente, `served_version` y `recommended_run_id` del catálogo servido |
| `GET /catalog` | Catálogo de runs |
| `GET /runs/{run_id}` | Manifiesto del run |
| `POST /evaluate` | `{run_id, level_id, seeds}`; devuelve métricas, casos y cuadros |
| `POST /sim/reset` | Crea una sesión de simulación |
| `POST /sim/step` | Ejecuta una acción manual o por regla en una sesión |
| `GET /monitor` | Solicitudes, errores y latencia por ruta (monitoreo) |

Detalle de cuerpos, respuestas y errores en [`despliegue-api.md`](despliegue-api.md).
Los baselines `baseline-random` y `baseline-rule` aparecen en el catálogo con
`level_id: "todos"` porque sirven en cualquier nivel.

## Cuadro de episodio

Cada cuadro mantiene el formato consumido por la vista 3D:

```text
[x, y, z, qw, qx, qy, qz, etapa, bateria, accion, recompensa, choque, [9 rayos]]
```

## Evaluación

`pipelines/evaluate.py` evalúa cualquier política congelada `policy(obs) -> acción`:

- **Suites fijas:** `configs/suites/suite-n1-v1.yaml`, `suite-n2-v1.yaml` y
  `suite-n3-v1.yaml`, con 50 episodios cada una. Las semillas empiezan en
  100000, 200000 y 300000, lejos de las de entrenamiento.
- **Test reservado:** `test-final` no guarda semillas en Git. Se leen de
  `EXPEDITION_TEST_SEEDS` o de `configs/suites/test-final.local.yaml`, que está
  ignorado. Se usa una sola vez por finalista.
- **Resultado (JSON):**
  - Identifica la evaluación: `suite_id`, `suite_sha256`, `level_id`,
    `environment_version`, `environment_hash`, `git_commit`, `git_dirty` y
    `policy` (con `model_sha256` en los modelos aprendidos).
  - `metrics`: `success_rate`, `failures_by_cause`, `route_efficiency`,
    `collisions_per_episode`, `final_battery_success`, `steps_mean`,
    `return_mean`/`return_std` y `latency_ms_mean`/`p95`.
  - `cases`: un registro por episodio.
  - En suites variadas, cada caso incluye `scenario_id`, `difficulty`,
    `beacons` y `active_rocks`; las métricas separan cobertura y éxito por
    dificultad.
- **Políticas** (`pipelines/policies.py`): `RandomPolicy`, `RulePolicy`,
  `QTablePolicy(.npz)` y `PPOPolicy(model.zip, vecnormalize.pkl)`. La API y el
  pipeline usan estas mismas clases para que la evaluación sea idéntica.
