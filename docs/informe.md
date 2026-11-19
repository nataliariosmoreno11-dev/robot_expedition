# Informe final · Robot Expedition

**Equipo:** Suleily Moreno, Tomás González, Natalia Ríos y Nicol Cerinza.
**Curso:** MLOps, Universidad Externado de Colombia, noviembre de 2026.

## 1. Problema y misión

Un rover con tracción diferencial, en un mundo 3D simulado con MuJoCo
(gravedad de Marte, dunas, grieta, rocas y arena), debe hacer lo siguiente:

1. salir de la base;
2. visitar las balizas en orden;
3. volver antes de agotar la batería.

Hay tres niveles con el mismo contrato de observación (20 floats) y de
acciones (`Discrete(5)`):

| Nivel | Contenido | Decisión nueva |
|---|---|---|
| N1 | 1 baliza, 2 rocas, sin dunas | Orientarse y rodear un obstáculo |
| N2 | 2 balizas, 5 rocas, dunas | Secuenciar dos objetivos y administrar la batería |
| N3 | N2 + bloqueo que aparece a mitad de la misión + sensor degradado | Replanificar |

El detalle está en la [ficha v1](ambiente-v1.md).

## 2. Decisiones de diseño

| Decisión | Por qué |
|---|---|
| La API Python es la única autoridad de la física | El enunciado advierte que exportar la red al navegador no traslada el ambiente. Así, un modelo histórico se evalúa con sus reglas originales; una prueba compara la API con la suite offline caso por caso |
| La etapa 3 significa siempre "volver a la base" | Un modelo entrenado en un nivel se puede evaluar en otro sin adaptador |
| Las suites fijas tienen semillas lejanas a las de entrenamiento y un test reservado fuera de Git | Separar el ajuste de la evaluación final |
| Hay un manifiesto por run con SHA-256 de modelo, config, ambiente y suite | La API se niega a servir pesos que no coinciden (409) |
| El catálogo está versionado, con historial y pin de rollback | Repetir una publicación no crea versión; un rollback no se deshace solo |
| Los criterios de selección se fijaron antes de los resultados (`configs/selection.yaml`) | Evitar elegir el criterio según el resultado |
| Los modelos se publican en una release (ZIP con SHA256SUMS), además de DVC | El remoto DVC de DagsHub es privado. El docente necesita recuperarlos sin cuenta |

## 3. Resultados por nivel

Éxito en las suites fijas de 50 episodios; media ± desviación estándar entre
semillas. Detalle en [experimentos.md](experimentos.md).

| | N1 | N2 | N3 |
|---|---|---|---|
| Aleatorio | 0 % | 0 % | 0 % |
| Regla | 100 % | 100 % | 98 % |
| Q-learning (exp-001, 3 semillas) | 39 ± 48 % | 97 ± 5 % | 50 ± 23 % |
| PPO (exp-002, 3 semillas) | 100 ± 0 % | 100 ± 0 % | 0 ± 0 % |
| PPO entrenado en N3 (exp-005, 3 semillas) | 50 ± 42 % | 44 ± 48 % | 100 ± 0 % |
| **Recomendado servido** | **100 %** (`a1752d56`) | **100 %** (`92cdc692`) | **100 %** (`3dd6e0f7`) |

**Finalistas** (exp-001 contra exp-002, 3 semillas cada uno). En el test
reservado:
- Q-learning: 98 / 100 / 100 %.
- PPO: 100 / 100 / 100 %.

PPO no es significativamente más exitoso, pero sí más eficiente (unos 85 pasos
frente a 117–195) y estable entre semillas. Se selecciona PPO y Q-learning
queda como baseline aprendido.

## 4. Incidente

Ver [incidente.md](incidente.md).
- **Medición:** se tomó el recomendado de N2 y se midió con las mismas
  50 semillas.
  - Su éxito cae de **100 % a 0 %** en N3.
  - Separando las causas, **el bloqueo explica toda la caída**; el sensor
    degradado solo no la cambia.
  - El rover gira frente a la roca hasta agotar la batería.
- **Intervención:** entrenar en N3 (exp-005, por el pipeline de Prefect).
  - Recupera el 100 % en N3.
  - Pero cae a 44 ± 48 % en N2.
- **Decisión con los criterios fijados:** se acepta para N3 y se rechaza para
  N2. Se sirve un modelo por nivel.

**Rollback** ([evidencia](evidencia/rollback.md)):
1. Una promoción manual errónea (`catalog-v3`) bajó el éxito servido en N2 a
   16 %.
2. El flow de rollback restauró el run de `catalog-v2` (`catalog-v4`).
3. La API confirmó la versión servida (`in_sync: true`) y el 100 %.

## 5. Costo

| Concepto | Valor |
|---|---|
| Entrenamiento | 12 runs, **125 min de CPU** en un portátil de 12 hilos, sin GPU. PPO: unos 13 min por run de 300k pasos. Q-learning: unos 9 min |
| Evaluación | 50 episodios en unos 15–30 s por modelo y nivel |
| Inferencia | 0.03 ms (Q-learning) y 0.25–0.4 ms (PPO) por decisión, con decisiones a 2 Hz |
| Servicios | Gratuitos: Vercel, Render (plan gratuito), GitHub Releases y DagsHub |

## 6. Límites

- **Mundo fijo:** balizas, base, grieta y dunas son fijas, y las semillas solo
  cambian el ruido. Un 100 % significa dominar *este* mapa, no navegar en
  general.
- **Techo:** en N2 varios candidatos llegan al 100 %. Con criterios de +5
  puntos, la selección no distingue entre ellos (exp-004 quedó recomendado
  por publicarse primero). Hace falta una suite más difícil o una métrica
  secundaria en la selección.
- **Sin repeticiones:** exp-003 (currículo) y exp-004 tienen una sola semilla.
  Son evidencia exploratoria.
- **Grieta invisible:** los rayos no ven la grieta. Los modelos de N2 que no
  son PPO caen en ella en N1.
- **MLflow sin interfaz pública:** el MLflow del equipo es local y se publica
  como archivo en la release. `scripts/mlflow_sync.py` lo copia a DagsHub, pero
  falta ejecutarlo con el token de Tomás.

## 7. Siguiente hipótesis razonable

> **Entrenar PPO con el bloqueo en el 30 % de los episodios** (y con el sensor
> degradado) produce un único modelo con éxito de al menos 95 % en N2 y N3.

Se compararía contra exp-002 (N2) y exp-005 (N3), con 300k pasos y 3 semillas.
Si falla, la alternativa es dar a la observación un "obstáculo en el camino al
objetivo" (un rayo dirigido al objetivo) en lugar de solo los 9 rayos fijos.

## 8. Segunda fase: exploración planetaria (releases `v1.2` y `v1.3`)

Las secciones 1 a 7 describen la entrega `v1.0` (ambiente de balizas, N1-N3), que
sigue disponible en el catálogo con sus contratos originales. La segunda fase
agrega un ambiente nuevo y, por tanto, una **familia de modelos nueva**: las
observaciones y las acciones cambian de tamaño y significado.

**Misión.** El rover sale de la base, encuentra con su cámara las muestras del tipo
pedido, evita señuelos, recolecta y vuelve a la base antes de agotar la batería.
Hay 8 escenarios fijos de dificultad creciente (`configs/campaign/`), con rocas que
forman corredores, cráteres, dunas, zonas de baja tracción, viento, polvo y tormentas.

| Contrato | Valor |
|---|---|
| Observación | 33 componentes × 4 lecturas apiladas = 132 (LIDAR 9 rayos 0-5 m, rumbo y distancia al objetivo visible, velocidad, giro, batería, inventario, tipo pedido y visto, gravedad, tracción, visibilidad, viento, cabeceo, alabeo, nivel) |
| Acciones | `Discrete(7)`: avanzar, dos arcos, girar a cada lado, recolectar, retroceder; decisión cada 0,5 s |
| Éxito (`terminated`) | Todas las muestras pedidas en la base con batería > 0 |
| Fallo (`terminated`) | Grieta o cráter (`z < −0,15 m`), vuelco, batería agotada, salida del área |
| Truncamiento | Límite de pasos del escenario (650-1.700); en entrenamiento, además, 120 pasos sin explorar ni recolectar |
| Recompensa | +0,12 por celda nueva explorada, +recolección correcta, −10,2 y −6 % de batería por señuelo, −penalización por paso, choque y atasco, +250 por misión completa (+15 × batería restante), −50 por grieta, vuelco o batería agotada, −35 por salir del área |

Manifiestos de todos los modelos de esta familia: [`manifests/planetary/`](../manifests/planetary)
(generados y verificados con `scripts/build_planetary_manifests.py`).

**PPO de campaña (exp-007 a exp-015).** Un escenario solo se aprueba con ≥ 50
misiones evaluadas, ≥ 85 % de éxito, ≤ 2 choques de media y retención de los
anteriores.

- **No mejoraron lo suficiente:** exp-008 a exp-013 (más exploración, retención,
  normalización fija, potencial observable, práctica por etapas). Ninguno aprobó el
  escenario 6: tras 5,27 M de pasos, 0/8 éxitos y 218 choques de media
  ([specialist-training.md](specialist-training.md)).
- **Mejoró:** exp-014/015 (un actor por escenario más imitación acotada). Aprobó
  los 8 escenarios con 7,46 M de pasos. Los hitos son `f7c104a3` (L1) a `4bcfd8f6`
  (L1-L8), y el modelo final servido es `f5972637` (7,49 M de pasos, física sin
  tormentas dinámicas, la de su entrenamiento).

**Q-learning planetario.** Q-learning tabular con un guía.

- **`seed-v1`:** solo resolvía L1 (~70 %).
- **Diagnóstico:** el guía tampoco resolvía L2-L8, los límites de 50/100 pasos
  hacían imposibles las misiones, los sensores llegaban con 1,5 s de atraso y había
  peligros invisibles.
- **`seed-v2`:** guía planificador (mapa de misión estático, sin muestras) y la
  acción del guía como parte del estado. La discretización pierde posición y
  memoria: el 32 % de las decisiones del guía eran indistinguibles con el estado
  v1.

Ver [auditoria-qlearning-planetario.md](auditoria-qlearning-planetario.md).

| Política (8 niveles, límites de diseño) | Éxito | Fuente |
|---|---|---|
| Aleatoria | 0 % | [auditoria-aprendizaje-2026-11-18.md](auditoria-aprendizaje-2026-11-18.md) |
| Guía reactivo anterior | 3/24 (solo L1) | auditoría Q-learning |
| **Guía planificador (regla)** | **79/80** | [evidencia](evidencia/evaluacion-planificador-qlearning.json) |
| Q-learning `seed-v1` | L1 ~70 %, L2-L8 0 % | auditoría Q-learning |
| **Q-learning `seed-v2` (recomendado)** | **77/80** (75/80 antes del regreso guiado) | [evidencia](evidencia/evaluacion-planificador-qlearning.json) |
| PPO de campaña `4bcfd8f6` / `f5972637` | 8/8 escenarios aprobados (≥ 50 misiones c/u) | `models/<run>/evidence/` |

**Límites de esta fase.**

- **Mapas memorizados:** el PPO aprueba los 8 mapas fijos, pero en mapas
  procedurales logró 50 % en L2 y 0 % en el resto (prueba pequeña), es decir,
  memoriza los mapas.
- **Guía con mapa:** el Q-learning depende de un guía que conoce el mapa estático,
  el equivalente a imágenes orbitales.
- **Aprendizaje en vivo no persistente:** Render gratuito no conserva lo aprendido
  en vivo; la API vuelve al checkpoint base en cada reinicio.
- **Rendimiento:** en la web cada petición cuesta unos 0,47 s de red. Por eso la API
  avanza hasta 40 pasos por petición.

**Siguiente hipótesis.** Entrenar el PPO en mapas procedurales (exp-016 a
exp-018), con la misma suite reservada, para medir generalización en vez de
memorización.

## 9. Contribuciones

| Integrante | Aportes |
|---|---|
| **Suleily Moreno** | Base del repositorio y primer ambiente (bípedo y terreno marciano); paso 1: niveles N1, N2 y N3 en YAML, `environment_hash`, contratos (`docs/contratos.md`) y dependencias fijadas (PR #2) |
| **Tomás González** | Paso 3: pipeline de entrenamiento trazable (`pipelines/train.py`: MLflow, manifiestos, checkpoints y continuación con `parent_run_id`); remoto DVC en DagsHub; release pública `models-v1` (PR #3) |
| **Natalia Ríos** | Paso 5: web Vite + TypeScript + Three.js (escena 3D, HUD, control manual y replay rotulado) y despliegue en Vercel (PR #4 y #5) |
| **Nicol Cerinza** | Ambiente 3D del rover en MuJoCo y ficha v0.1; integración de las ramas; paso 2 (suite fija y baselines); paso 4 (API); pasos 6–11 (experimentos, ficha v1, Prefect, catálogo y selección, vista del docente, incidente y rollback); documentación final |

**Segunda fase (`v1.0` → `v1.3`):**

| Integrante | Aportes |
|---|---|
| **Natalia Ríos** | Simulador planetario y escenarios variados; Q-learning planetario persistente con checkpoints reanudables y regreso guiado a la base; progresión por nivel, campaña PPO y visor PPO en la web; despliegue en Vercel y Render |
| **Tomás González** | Campaña fija de escenarios; entrenamiento PPO versionado (exp-007 a exp-015), actores por escenario e imitación acotada; hitos aprobados L1-L8 y modelo final `f5972637`; panel de entrenamiento PPO en la web |
| **Nicol Cerinza** | Auditoría y rediseño del Q-learning planetario (guía planificador, estado v2, `qlearning-seed-v1/v2`); progreso de campaña en la API; pasos por lotes; manifiestos planetarios; releases `v1.2` y `v1.3` y documentación |
| **Suleily Moreno** | Revisión de la entrega con el enunciado; preparación y presentación de la sustentación: problema y contexto de negocio, costo, limitaciones y siguiente hipótesis, y demostración ([guion-defensa.md](guion-defensa.md)) |

**Uso de asistentes:** parte del código y de los documentos se escribió con
asistentes de código (Copilot y Claude Code). Cada afirmación de este informe
sale de un archivo del repositorio (run_id, JSON de resultados o evidencia), y
todos los integrantes deben poder explicarlas
([guion-defensa.md](guion-defensa.md)).
