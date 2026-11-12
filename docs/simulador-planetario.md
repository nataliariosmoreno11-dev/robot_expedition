# Simulador de exploración planetaria

## Alcance implementado

El simulador agrega una misión procedural de recolección sobre Marte sin reemplazar
el ambiente anterior. Cada episodio varía terreno, alturas, rocas, cráteres, dunas,
muestras objetivo y distractoras, viento, polvo, temperatura, tracción, batería y
posición de los objetivos. Los ocho niveles aumentan tamaño útil, cantidad de
muestras, relieve, obstáculos y perturbaciones.

El currículo se basa en desempeño. Mantiene una ventana de éxitos, avanza cuando
alcanza el umbral configurado y mezcla niveles anteriores para reducir olvido. Su
estado se escribe en JSON junto al entrenamiento y se copia al checkpoint al
reanudar.

## Probar una sesión manual

```bash
python -m uvicorn api.main:app --reload --port 8000
```

En otra terminal:

```bash
curl -X POST http://127.0.0.1:8000/sim/reset \
  -H 'Content-Type: application/json' \
  -d '{"environment":"planetary","difficulty_level":3,"seed":42}'
```

Con el `session_id` devuelto se envían movimientos y la acción 5 para recolectar:

```bash
curl -X POST http://127.0.0.1:8000/sim/step \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"...","action":5}'
```

La web permite elegir L1–L8, conducir, recolectar, consultar intentos guardados y
reproducir uno de ellos. Las muestras turquesa son del tipo solicitado y las ocres
son distractoras. Las muestras recogidas desaparecen del mapa durante el replay.

## Consultar episodios sin abrir la web

```bash
python scripts/episodes.py list
python scripts/episodes.py show 12
python scripts/episodes.py replay 12 --output /tmp/episode-12.json
python scripts/episodes.py scenario 12
python scripts/episodes.py compare 10 11 12
```

Puede elegirse otra base anteponiendo `--db ruta.sqlite` al subcomando, por ejemplo
`python scripts/episodes.py --db /tmp/prueba.sqlite list`.

## Entrenamiento y evaluación

El experimento de referencia está en
[`configs/experiments/exp-007-planetary.yaml`](../configs/experiments/exp-007-planetary.yaml):

```bash
python pipelines/train.py --config configs/experiments/exp-007-planetary.yaml --seed 1
```

PPO recibe 33 observaciones y seis acciones. El entrenamiento registra dificultad,
éxito y muestras. Cada 100.000 pasos ejecuta una evaluación determinista sobre el
holdout fijo que comienza en la semilla 910.000 y guarda
`periodic-evaluation-<paso>.json`; la evaluación final usa otro conjunto reservado
que comienza en 900.000. Ninguna de las dos actualiza el currículo. Se reportan
éxito por dificultad, causas de fallo, muestras, recorrido, choques, pasos,
recompensa y latencia. No se incluye un checkpoint fingido: producir curvas y
métricas finales requiere ejecutar el presupuesto real.

## Qué representa la física

- MuJoCo aplica gravedad, altura, contactos, fricción y fuerzas externas.
- El viento usa densidad atmosférica marciana, coeficiente de arrastre, área frontal
  y ráfagas acotadas.
- Polvo limita el alcance y aumenta ruido de los rayos.
- Frío, baja tracción, movimiento y viento modifican el consumo de batería.
- Pendiente y orientación llegan a la observación; el vuelco es una causa real de fin.

El terreno sigue siendo un heightfield y las rocas son cuerpos geométricos simples.
No modela suelo granular, electrostática del polvo, transferencia térmica detallada
ni aerodinámica CFD. Estas simplificaciones mantienen el entrenamiento reproducible
y ejecutable en un computador de clase.

## Validación

```bash
python -m pytest -q
cd web && npm ci && npm run build
```

Las pruebas cubren reproducibilidad y variación entre semillas, solvencia de L1–L8,
recolección correcta/incorrecta/repetida, parámetros físicos, choques, avance del
currículo, no alteración por inspección, éxito completo, persistencia incremental,
interrupciones y replay inmutable.
