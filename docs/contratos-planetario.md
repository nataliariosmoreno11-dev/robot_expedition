# Contrato planetario v2

Versión: `planetary-exploration/2.0.0`. El ambiente anterior de balizas conserva
20 observaciones y cinco acciones. No se pueden intercambiar sus checkpoints.

## Escenarios fijos

`scenario_01` a `scenario_10` identifican las diez definiciones en
`configs/campaign/`. El hash del catálogo forma parte del hash del ambiente.
La semilla del episodio controla sensores; geometría, objetivos, terreno, clima y
presupuestos se cargan de la definición. Una misión termina con muestras completas
y regreso a la base. Los escenarios 06–10 requieren muestras de dos tipos.

## Acciones

| Índice | Acción |
|---:|---|
| 0 | avanzar |
| 1 | avanzar + izquierda |
| 2 | avanzar + derecha |
| 3 | girar izquierda |
| 4 | girar derecha |
| 5 | recolectar |
| 6 | retroceder |

`recolectar` solo funciona dentro del radio de colección. Una muestra del tipo
incorrecto, una repetida, un inventario lleno o intentar recolectar sin una muestra
cercana tienen penalizaciones distintas. La misión solo termina con éxito después
de reunir el número solicitado del tipo correcto y regresar a la base.

## Observación de 33 valores

| Índices | Contenido | Escala |
|---|---|---|
| 0–8 | nueve rayos de distancia | metros, limitados por visibilidad |
| 9 | distancia a objetivo visible (o alcance máximo) | metros |
| 10–11 | rumbo relativo visible; ambos cero si es desconocido | `[-1, 1]` |
| 12–13 | velocidad y giro | normalizados |
| 14 | batería | `[0, 1]` |
| 15–17 | fracción recogida, restante e inventario | `[0, 1]` |
| 18–20 | tipo solicitado: roca, hielo, mineral | multi-hot de tipos pendientes |
| 21–22 | muestra visible y distancia | indicador, metros |
| 23–25 | tipo de muestra visible | one-hot |
| 26–29 | gravedad, tracción, visibilidad y viento | normalizados |
| 30–31 | inclinación longitudinal y lateral | radianes |
| 32 | dificultad curricular | nivel / 10 |

La especificación del manifiesto guarda tamaño de observación y acciones. Un modelo
del contrato de balizas no puede servirse dentro de una sesión planetaria.
Las posiciones de muestras ocultas no forman parte de la observación. El detector
solo informa una muestra dentro del alcance afectado por polvo y con línea de visión
sin rocas; después de completar el inventario, la base sí se usa como destino conocido.

## Memoria y versiones

Cada observación instantánea tiene 33 valores. `history_steps: 4` concatena las
últimas cuatro (132 floats); al reiniciar se repite la observación inicial.
PPO sigue siendo MlpPolicy. La memoria y la normalización se mantienen durante
el entrenamiento; el checkpoint guarda optimizador, pesos, normalización y contador.
No hay rutas del planificador ni posiciones ocultas en la observación.

## Persistencia

El esquema SQLite 2 se amplía de forma aditiva con `analysis_json`. Cada paso se
confirma en WAL y su estado se comprime con zlib. Se guardan pose, observación,
acción, recompensa, inventario, eventos, recursos y versión de política. El
registro de campaña no poda episodios, aunque el constructor reciba un límite
antiguo. Los lectores respetan los registros históricos existentes.

Un episodio contiene `scenario_id`, semilla sensorial, definición del escenario,
versión de política, modo y pasos. Los modos se distinguen: `entrenamiento`,
`evaluación`, `robustez`, `manual`. Las trayectorias se cargan solo al consultar
un episodio, nunca se mantienen todas en memoria. Reinicio y cierre normal
finalizan el intento; tras un cierre abrupto los pasos confirmados siguen en SQLite.

## API

```text
GET  /scenarios
GET  /scenarios/{scenario_id}/analysis
GET  /scenarios/{scenario_id}/history?mode=entrenamiento&limit=100&offset=0
GET  /scenarios/{scenario_id}/progress?mode=entrenamiento
GET  /campaign/progress
POST /sim/reset  {environment: planetary, difficulty_level: 1..10, seed, previous_session_id?}
POST /sim/step   {session_id, action: 0..6}
GET  /episodes?scenario_id=scenario_08&mode=entrenamiento&limit=100&offset=0
GET  /episodes/{number}
GET  /episodes/{number}/replay
GET  /episodes/{number}/scenario
POST /episodes/{number}/protect
POST /episodes/compare {episode_numbers: [100,500,1000]}
GET  /episodes/metrics/summary
```

`previous_session_id` cierra el intento anterior al iniciar uno nuevo. Los replays
incluyen `reference_path` para análisis; la web lo mantiene oculto en control manual.
`path_efficiency` es nulo para fallos o costes no comparables. Las referencias son
óptimas solo en la rejilla de validación y sobre las posiciones de interacción
comparadas; no constituyen un óptimo continuo ni una política aprendida.

La ruta de SQLite se controla con `EXPEDITION_EPISODE_DB`. Los umbrales del currículo
se configuran en `environment.curriculum_settings`; el estado de cada run se guarda
en `models/<run_id>/curriculum_state.json`.
