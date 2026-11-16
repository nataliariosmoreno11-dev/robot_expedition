# Recuperaci?n del nivel 6 mediante PPO especializado e imitaci?n

El ciclo anterior lleg? a 5.266.272 pasos sin aprobar el nivel 6. La ?ltima evaluaci?n
mostr? 0/8 ?xitos, 2,50 muestras y 217,75 choques de media. La retenci?n rechazaba
las actualizaciones porque da?aban niveles anteriores y restauraba los pesos seguros.
Los pasos contabilizan trabajo realizado, no avance consolidado; no se borran.

## Pol?tica

Se mantiene PPO. Cada escenario tiene una copia independiente del actor. Los actores aprobados
quedan congelados y PPO actualiza los pendientes. Una reparaci?n supervisada
expl?cita puede corregir un actor anterior, pero exige una nueva validaci?n.
La selecci?n usa la dificultad ya presente en la observaci?n. VecNormalize permanece
congelado. El cr?tico sigue aprendiendo; el optimizador se reinicia al cambiar de
arquitectura. Las ejecuciones nuevas desde cero no congelan un actor aleatorio:
la separaci?n solo se aplica cuando el padre ya tiene los cinco niveles aprobados.

El actor del nivel 6 recibe inicializaci?n por clonaci?n de comportamiento:
369 pares observaci?n/acci?n de una demostraci?n f?sica correcta. El nivel 2
se repara con otros 124 pares; la regresi?n heredada pas? de 456 choques a cero. El planificador
conoce el escenario solo para generar esas etiquetas de entrenamiento. La pol?tica
exportada recibe ?nicamente sus 132 observaciones; no llama al planificador.
SARSA no se incorpor?: el cambio probado mantiene PPO y aborda el olvido y la falta
de trayectorias exitosas sin sustituir el contrato de observaci?n ni el evaluador.

## Experimento acotado

`pipelines/specialist_policy.py` contiene ScenarioPolicy, serializable con SB3.
Conserva tambi?n la clase del primer piloto para poder leer sus archivos.
`configs/experiments/exp-014-specialist.yaml` define la continuaci?n con PPO.

```sh
.venv/bin/python scripts/training_control.py stop
# Esperar a que status confirme DETENIDO y el checkpoint est? completo.
.venv/bin/python scripts/pilot_scenarios.py --parent models/ID_PADRE --output logs/piloto-nuevo --levels 2 6
.venv/bin/python scripts/validate_specialist.py --parent models/ID_PADRE --candidate logs/piloto-nuevo/candidate.zip --output logs/validacion-nueva
```

El piloto compara 0?2.500 actualizaciones supervisadas, en tramos de 500, con tres
semillas de diagn?stico 740000?740002 y la regresi?n conocida 920047 del nivel 2.
Esta regresi?n es dato de desarrollo, no evidencia independiente. Conserva un
candidato por ejecuci?n, elegido por ?xito, muestras, menos choques y menos pasos. Estas pruebas NO aprueban el curr?culo.
La validaci?n usa 50 semillas diferentes del nivel 6 y 50 de cada nivel 1?5,
con semillas disjuntas (930000 para el 6, 940000 para retenci?n),
con los criterios oficiales sin cambios: ?xito m?nimo 85%, choques medios m?ximo 2.
Solo exporta un checkpoint y un trabajo completado si todas las suites pasan.
No modifica la campa?a en ejecuci?n ni la reanuda autom?ticamente.

El checkpoint exportado conserva el contador de interacci?n anterior y registra
por separado las actualizaciones de imitaci?n. No inventa pasos PPO para ganar
prioridad al reanudar. `resume` prioriza el contador; si empata, selecciona el checkpoint con m?s
escenarios aprobados y despu?s el guardado m?s reciente. As? reconoce la mejora
por imitaci?n sin inventar pasos de interacci?n.

## Almacenamiento y l?mites

La demostraci?n del nivel 6 ocupa unos 25 KB, m?s el peque?o archivo del nivel 2. Solo se guarda un candidato de piloto,
los informes y, si aprueba, una versi?n completa. PPO conserva los l?mites existentes
de dos checkpoints intermedios y uno rechazado por ejecuci?n; las versiones finales
siguen acumul?ndose. No se borra el punto seguro ni el historial.

Pasar la campa?a fija no demuestra generalizaci?n a mapas nuevos. Las pruebas de
robustez se consideran separadas. Al aprobar el nivel 6 se congela su actor; aprender el 7 no modifica sus pesos.
Las evaluaciones de retenci?n siguen siendo obligatorias.

Se reutiliza una cuadr?cula de referencia por escenario durante la evaluaci?n.
Se reutiliza tambi?n el coste de referencia cuando las posiciones son exactamente
iguales. Las pruebas verifican rutas/costes id?nticos y que cada episodio se simula
con su semilla; no se reutilizan resultados de episodios ni se reducen los intentos.
