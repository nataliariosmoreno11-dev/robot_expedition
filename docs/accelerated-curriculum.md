# Entrenamiento acelerado de la campaña

Configuración: `configs/experiments/exp-015-accelerated.yaml`.

Al iniciar una continuación con las políticas independientes, el entrenador intenta
aprobar el escenario pendiente mediante un ciclo acotado:

1. Genera cuatro demostraciones físicas con semillas de entrenamiento distintas.
   Rechaza cualquier demostración que no complete la misión o supere dos choques.
2. Divide los ejemplos por número de muestras recogidas. El Último fragmento es
   el regreso a la base. Muestrea los fragmentos de manera equilibrada para que
   la exploración inicial no oculte el aprendizaje del regreso.
3. Realiza hasta 2.000 actualizaciones supervisadas. Cada 500 ejecuta tres misiones
   diagnósticas desde la base y conserva la mejor política, no necesariamente la última.
4. Si el diagnóstico es bueno, prueba la suite oficial de 50 misiones completas;
   también valida 50 misiones de cada escenario aprobado. Se mantienen ≥85% de
   éxitos y ≤2 choques medios. Los diagnósticos y los fragmentos no aprueban niveles.
5. Si pasa, congela el actor, guarda checkpoint y un replay del modelo aprendido,
   y aplica el mismo proceso al siguiente escenario dentro de la misma ejecución.
6. Si no pasa, continúa PPO. La mitad de los reinicios del escenario puede comenzar
   en estados físicos intermedios de las demostraciones; conservan batería gastada
   y pasos transcurridos. Las evaluaciones siempre comienzan en la base.

Se permiten dos intentos de inicialización por escenario entre continuaciones. El
historial viaja con el checkpoint para evitar repetir indefinidamente el mismo piloto.
No hay garantía de tiempo ni de aprobación; un escenario difícil puede necesitar
cambios adicionales si ni el piloto ni PPO progresan.

La política recibe los sensores habituales; el planificador solo genera etiquetas
para entrenamiento. La batería, el mapa y los criterios de evaluación no cambian.
Las actualizaciones supervisadas se registran aparte de los pasos de interacción.

## Seguimiento

La página muestra la fase, las actualizaciones supervisadas, el avance de las
validaciones y la coincidencia con ejemplos de cada fragmento. Esa coincidencia
no equivale a una tasa de éxito en misiones. Desde la terminal:

```sh
.venv/bin/python scripts/training_control.py watch
```

`models/<run>/acceleration.json` contiene el estado actual y
`acceleration-history.json` conserva los intentos y resultados por escenario.
Los informes `accelerated-*.json` contienen los casos de evaluación. Los checkpoints
`mastered-scenario-*` conservan cada promoción.

## Almacenamiento y velocidad

Cuatro demostraciones se comprimen en un archivo por escenario; los archivos
transitorios se eliminan al combinarlos. El mejor candidato se mantiene en memoria.
Se conserva un checkpoint por escenario aprobado y se mantienen los límites de
PPO de dos checkpoints intermedios y uno rechazado por ejecución. Las versiones
finales siguen acumulándose; no se afirma un límite global de almacenamiento.
El cálculo de referencias reutiliza cuadrículas y posiciones idénticas, pero cada
misión de evaluación se simula de nuevo. La inferencia de un solo rover ejecuta
únicamente el actor correspondiente al escenario.
