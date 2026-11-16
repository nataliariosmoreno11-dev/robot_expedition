# Experimento 013: pr?ctica por tramos del escenario 6

## Diagn?stico reproducido

Pesos seguros `ef2bcbedc60745e396b33f46e3b0bc1a`, semilla 910000, misi?n
completa sin ayudas: segunda muestra en paso 137, tercera en 1077; termin?
sin bater?a en 1292 pasos. Ejecut? retroceso 721 veces. Los principales focos
de choques fueron alrededor de (5,-1) y (5,-2) con dos muestras, y (7,3),
(7,2) con tres. La pol?tica pierde tiempo navegando entre muestras; aumentar
la bater?a esconder?a el problema.

## Intervenci?n

`configs/experiments/exp-013-stage-practice.yaml` contin?a desde los pesos
verificados a 4.483.936 pasos. Mantiene el repaso de 1?5 y la protecci?n contra
olvido. La mitad de los reinicios del nivel 6 parte de la base y la otra mitad
elige uniformemente tres estados: tras dos muestras, tras tres, e inventario
completo para practicar el regreso. Esto es mezcla de etapas, no aprobaci?n
secuencial autom?tica. PPO elige todas las acciones despu?s del reinicio.

`scripts/build_stage_bank.py` genera esos estados mediante una demostraci?n
con conocimiento del mapa. Es conocimiento privilegiado SOLO para preparar
inicios de entrenamiento; no se entrega al modelo ni se usa en evaluaci?n.
La demostraci?n complet? la misi?n en 369 pasos, cero choques y 59,37 % de
bater?a restante. **Esto no es un resultado del agente aprendido.**

El banco conserva posici?n y velocidad f?sicas, tiempo transcurrido, bater?a
gastada, inventario, historial de observaciones y cobertura. No repone bater?a
ni reinicia el presupuesto de pasos. Solo ocupa unos 9 KB. Est? versionado y
valida el hash del cat?logo y la compatibilidad del historial de observaciones.
Si cambia el mapa hay que regenerarlo expl?citamente.

## Separaci?n de resultados

Los replays parciales llevan modo `entrenamiento_etapa`; sus resultados no
actualizan el curr?culo. La m?trica es `training/stage_success_rate`, separada
de `training/success_rate`. Evaluaci?n y modo manual ignoran el banco incluso
si el archivo no existe: siempre empiezan en la base con inventario vac?o.
El nivel 6 solo se aprueba con la suite completa original, y contin?a el
control de conservaci?n de los escenarios anteriores.

No se ha aplicado clonaci?n de comportamiento ni se ha sustituido PPO por un
planificador. Si practicar estos tramos tampoco mejora las evaluaciones, el
siguiente experimento ser?a aprender acciones de demostraciones, incluyendo
repaso y evaluaci?n de retenci?n antes de adoptar los pesos.

## Almacenamiento y seguimiento

Se mantiene el l?mite de dos checkpoints intermedios y uno rechazado por
versi?n; la SQLite compartida conserva replays compactos. La campa?a se
pausa para guardar al bajar de 10 GiB libres en Linux o C:.

```sh
.venv/bin/python scripts/training_control.py watch
```

Comparar ?xito de misi?n completa, muestras y choques; no interpretar el
contador acumulado ni ?xitos parciales como aprobaci?n del 6.
