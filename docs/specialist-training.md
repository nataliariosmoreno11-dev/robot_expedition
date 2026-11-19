# Recuperación del nivel 6 mediante PPO especializado e imitación

El ciclo anterior llegó a 5.266.272 pasos sin aprobar el nivel 6. La última evaluación
mostró 0/8 éxitos, 2,50 muestras y 217,75 choques de media. La retención rechazaba
las actualizaciones porque dañaban niveles anteriores y restauraba los pesos seguros.
Los pasos contabilizan trabajo realizado, no avance consolidado; no se borran.

## Política

Se mantiene PPO. Cada escenario tiene una copia independiente del actor. Los actores aprobados
quedan congelados y PPO actualiza los pendientes. Una reparación supervisada
explícita puede corregir un actor anterior, pero exige una nueva validación.
La selección usa la dificultad ya presente en la observación. VecNormalize permanece
congelado. El crítico sigue aprendiendo; el optimizador se reinicia al cambiar de
arquitectura. Las ejecuciones nuevas desde cero no congelan un actor aleatorio:
la separación solo se aplica cuando el padre ya tiene los cinco niveles aprobados.

El actor del nivel 6 recibe inicialización por clonación de comportamiento:
369 pares observación/acción de una demostración física correcta. El nivel 2
se repara con otros 124 pares; la regresión heredada pasó de 456 choques a cero. El planificador
conoce el escenario solo para generar esas etiquetas de entrenamiento. La política
exportada recibe únicamente sus 132 observaciones; no llama al planificador.
SARSA no se incorporó: el cambio probado mantiene PPO y aborda el olvido y la falta
de trayectorias exitosas sin sustituir el contrato de observación ni el evaluador.

## Experimento acotado

`pipelines/specialist_policy.py` contiene ScenarioPolicy, serializable con SB3.
Conserva también la clase del primer piloto para poder leer sus archivos.
`configs/experiments/exp-014-specialist.yaml` define la continuación con PPO.

```sh
.venv/bin/python scripts/training_control.py stop
# Esperar a que status confirme DETENIDO y el checkpoint está completo.
.venv/bin/python scripts/pilot_scenarios.py --parent models/ID_PADRE --output logs/piloto-nuevo --levels 2 6
.venv/bin/python scripts/validate_specialist.py --parent models/ID_PADRE --candidate logs/piloto-nuevo/candidate.zip --output logs/validacion-nueva
```

El piloto compara 0–2.500 actualizaciones supervisadas, en tramos de 500, con tres
semillas de diagnóstico 740000–740002 y la regresión conocida 920047 del nivel 2.
Esta regresión es dato de desarrollo, no evidencia independiente. Conserva un
candidato por ejecución, elegido por éxito, muestras, menos choques y menos pasos. Estas pruebas NO aprueban el currículo.
La validación usa 50 semillas diferentes del nivel 6 y 50 de cada nivel 1–5,
con semillas disjuntas (930000 para el 6, 940000 para retención),
con los criterios oficiales sin cambios: éxito mínimo 85%, choques medios máximo 2.
Solo exporta un checkpoint y un trabajo completado si todas las suites pasan.
No modifica la campaña en ejecución ni la reanuda automáticamente.

El checkpoint exportado conserva el contador de interacción anterior y registra
por separado las actualizaciones de imitación. No inventa pasos PPO para ganar
prioridad al reanudar. `resume` prioriza el contador; si empata, selecciona el checkpoint con más
escenarios aprobados y después el guardado más reciente. Así reconoce la mejora
por imitación sin inventar pasos de interacción.

## Almacenamiento y límites

La demostración del nivel 6 ocupa unos 25 KB, más el pequeño archivo del nivel 2. Solo se guarda un candidato de piloto,
los informes y, si aprueba, una versión completa. PPO conserva los límites existentes
de dos checkpoints intermedios y uno rechazado por ejecución; las versiones finales
siguen acumulándose. No se borra el punto seguro ni el historial.

Pasar la campaña fija no demuestra generalización a mapas nuevos. Las pruebas de
robustez se consideran separadas. Al aprobar el nivel 6 se congela su actor; aprender el 7 no modifica sus pesos.
Las evaluaciones de retención siguen siendo obligatorias.

Se reutiliza una cuadrícula de referencia por escenario durante la evaluación.
Se reutiliza también el coste de referencia cuando las posiciones son exactamente
iguales. Las pruebas verifican rutas/costes idénticos y que cada episodio se simula
con su semilla; no se reutilizan resultados de episodios ni se reducen los intentos.
