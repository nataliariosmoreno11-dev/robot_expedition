# Experimento 013: práctica por tramos del escenario 6

## Diagnóstico reproducido

Pesos seguros `ef2bcbedc60745e396b33f46e3b0bc1a`, semilla 910000, misión
completa sin ayudas: segunda muestra en paso 137, tercera en 1077; terminó
sin batería en 1292 pasos. Ejecutó retroceso 721 veces. Los principales focos
de choques fueron alrededor de (5,-1) y (5,-2) con dos muestras, y (7,3),
(7,2) con tres. La política pierde tiempo navegando entre muestras; aumentar
la batería escondería el problema.

## Intervención

`configs/experiments/exp-013-stage-practice.yaml` continúa desde los pesos
verificados a 4.483.936 pasos. Mantiene el repaso de 1–5 y la protección contra
olvido. La mitad de los reinicios del nivel 6 parte de la base y la otra mitad
elige uniformemente tres estados: tras dos muestras, tras tres, e inventario
completo para practicar el regreso. Esto es mezcla de etapas, no aprobación
secuencial automática. PPO elige todas las acciones después del reinicio.

`scripts/build_stage_bank.py` genera esos estados mediante una demostración
con conocimiento del mapa. Es conocimiento privilegiado SOLO para preparar
inicios de entrenamiento; no se entrega al modelo ni se usa en evaluación.
La demostración completó la misión en 369 pasos, cero choques y 59,37 % de
batería restante. **Esto no es un resultado del agente aprendido.**

El banco conserva posición y velocidad físicas, tiempo transcurrido, batería
gastada, inventario, historial de observaciones y cobertura. No repone batería
ni reinicia el presupuesto de pasos. Solo ocupa unos 9 KB. Está versionado y
valida el hash del catálogo y la compatibilidad del historial de observaciones.
Si cambia el mapa hay que regenerarlo explícitamente.

## Separación de resultados

Los replays parciales llevan modo `entrenamiento_etapa`; sus resultados no
actualizan el currículo. La métrica es `training/stage_success_rate`, separada
de `training/success_rate`. Evaluación y modo manual ignoran el banco incluso
si el archivo no existe: siempre empiezan en la base con inventario vacío.
El nivel 6 solo se aprueba con la suite completa original, y continúa el
control de conservación de los escenarios anteriores.

No se ha aplicado clonación de comportamiento ni se ha sustituido PPO por un
planificador. Si practicar estos tramos tampoco mejora las evaluaciones, el
siguiente experimento sería aprender acciones de demostraciones, incluyendo
repaso y evaluación de retención antes de adoptar los pesos.

## Almacenamiento y seguimiento

Se mantiene el límite de dos checkpoints intermedios y uno rechazado por
versión; la SQLite compartida conserva replays compactos. La campaña se
pausa para guardar al bajar de 10 GiB libres en Linux o C:.

```sh
.venv/bin/python scripts/training_control.py watch
```

Comparar éxito de misión completa, muestras y choques; no interpretar el
contador acumulado ni éxitos parciales como aprobación del 6.
