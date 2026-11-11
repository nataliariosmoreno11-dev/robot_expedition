# Escenarios variados para generalización

El conjunto [`varied-v1`](../configs/scenarios/varied-v1.yaml) evita que una
política aprenda una sola secuencia de giros. En cada episodio cambia la
posición de B1 y B2, el número y la posición de las rocas y, en el escenario
más difícil, aparece un bloqueo después de visitar B1.

La observación sigue teniendo 20 valores y las acciones siguen siendo las
mismas cinco. El agente recibe la distancia y el ángulo relativos al objetivo
activo, por lo que puede aprender a navegar hacia coordenadas nuevas sin ver
`x, y` directamente.

## Escenarios

| Dificultad | Escenario | B1 → B2 | Rocas | Variación | Bloqueo |
|---|---|---|---:|---:|---|
| Fácil | `easy-open-north` | (4.0, −0.8) → (6.5, 2.8) | 1 | ±0.15 m | no |
| Fácil | `easy-open-south` | (4.0, −1.4) → (6.8, −3.6) | 2 | ±0.20 m | no |
| Medio | `medium-zigzag` | (7.2, 3.8) → (8.7, −3.5) | 4 | ±0.50 m | no |
| Medio | `medium-reverse` | (8.8, −3.8) → (7.2, 3.6) | 4 | ±0.60 m | no |
| Difícil | `hard-long-route` | (9.5, 3.5) → (9.5, −3.5) | 5 | ±0.80 m | no |
| Difícil | `hard-blocked-return` | (10.5, −4.2) → (10.5, 3.2) | 5 | ±1.00 m | después de B1 |

Las rocas solo aceptan una posición si dejan libres la base, las balizas y la
grieta, y si una búsqueda en rejilla confirma que la misión sigue siendo
alcanzable. La misma semilla reproduce exactamente el escenario y sus rocas.
Con semillas consecutivas, `seed % 6` recorre los seis escenarios; los resets
sin semilla continúan muestreándolos con el generador del ambiente.

## Entrenamiento

[`exp-006.yaml`](../configs/experiments/exp-006.yaml) activa `varied-v1` para
PPO. Los experimentos anteriores y la API de producción conservan sus mapas
fijos mientras no declaren `scenario_set`.

```bash
python pipelines/train.py --config configs/experiments/exp-006.yaml --seed 1
```

Ese entrenamiento requiere el presupuesto declarado de 300 000 pasos. El
archivo configura la suite de generalización como evaluación final, pero no
declara resultados hasta que el run termine y se registre en MLflow.

## Evaluación de generalización

La suite [`suite-n2-generalization-v1`](../configs/suites/suite-n2-generalization-v1.yaml)
usa 60 semillas fuera del entrenamiento: diez episodios por escenario.

```bash
python pipelines/evaluate.py \
  --policy ppo \
  --model models/<run_id>/ppo_model.zip \
  --vecnormalize models/<run_id>/vecnormalize.pkl \
  --level n2 \
  --suite suite-n2-generalization-v1
```

Además de las métricas normales, el resultado incluye
`episodes_by_difficulty`, `success_rate_by_difficulty` y, por caso,
`scenario_id`, `difficulty`, `beacons`, `active_rocks` y la ruta de referencia
correspondiente a ese mapa.
