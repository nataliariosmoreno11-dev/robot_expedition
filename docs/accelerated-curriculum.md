# Entrenamiento acelerado de la campa?a

Configuraci?n: `configs/experiments/exp-015-accelerated.yaml`.

Al iniciar una continuaci?n con las pol?ticas independientes, el entrenador intenta
aprobar el escenario pendiente mediante un ciclo acotado:

1. Genera cuatro demostraciones f?sicas con semillas de entrenamiento distintas.
   Rechaza cualquier demostraci?n que no complete la misi?n o supere dos choques.
2. Divide los ejemplos por n?mero de muestras recogidas. El ?ltimo fragmento es
   el regreso a la base. Muestrea los fragmentos de manera equilibrada para que
   la exploraci?n inicial no oculte el aprendizaje del regreso.
3. Realiza hasta 2.000 actualizaciones supervisadas. Cada 500 ejecuta tres misiones
   diagn?sticas desde la base y conserva la mejor pol?tica, no necesariamente la ?ltima.
4. Si el diagn?stico es bueno, prueba la suite oficial de 50 misiones completas;
   tambi?n valida 50 misiones de cada escenario aprobado. Se mantienen ?85% de
   ?xitos y ?2 choques medios. Los diagn?sticos y los fragmentos no aprueban niveles.
5. Si pasa, congela el actor, guarda checkpoint y un replay del modelo aprendido,
   y aplica el mismo proceso al siguiente escenario dentro de la misma ejecuci?n.
6. Si no pasa, contin?a PPO. La mitad de los reinicios del escenario puede comenzar
   en estados f?sicos intermedios de las demostraciones; conservan bater?a gastada
   y pasos transcurridos. Las evaluaciones siempre comienzan en la base.

Se permiten dos intentos de inicializaci?n por escenario entre continuaciones. El
historial viaja con el checkpoint para evitar repetir indefinidamente el mismo piloto.
No hay garant?a de tiempo ni de aprobaci?n; un escenario dif?cil puede necesitar
cambios adicionales si ni el piloto ni PPO progresan.

La pol?tica recibe los sensores habituales; el planificador solo genera etiquetas
para entrenamiento. La bater?a, el mapa y los criterios de evaluaci?n no cambian.
Las actualizaciones supervisadas se registran aparte de los pasos de interacci?n.

## Seguimiento

La p?gina muestra la fase, las actualizaciones supervisadas, el avance de las
validaciones y la coincidencia con ejemplos de cada fragmento. Esa coincidencia
no equivale a una tasa de ?xito en misiones. Desde la terminal:

```sh
.venv/bin/python scripts/training_control.py watch
```

`models/<run>/acceleration.json` contiene el estado actual y
`acceleration-history.json` conserva los intentos y resultados por escenario.
Los informes `accelerated-*.json` contienen los casos de evaluaci?n. Los checkpoints
`mastered-scenario-*` conservan cada promoci?n.

## Almacenamiento y velocidad

Cuatro demostraciones se comprimen en un archivo por escenario; los archivos
transitorios se eliminan al combinarlos. El mejor candidato se mantiene en memoria.
Se conserva un checkpoint por escenario aprobado y se mantienen los l?mites de
PPO de dos checkpoints intermedios y uno rechazado por ejecuci?n. Las versiones
finales siguen acumul?ndose; no se afirma un l?mite global de almacenamiento.
El c?lculo de referencias reutiliza cuadr?culas y posiciones id?nticas, pero cada
misi?n de evaluaci?n se simula de nuevo. La inferencia de un solo rover ejecuta
?nicamente el actor correspondiente al escenario.
