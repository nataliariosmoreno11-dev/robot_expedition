# Experimento 012: navegación observable

## Motivo y punto de partida

El experimento 011 siguió sin aprobar el escenario 6. En la evaluación de
4.137.824 pasos registró 0/8 éxitos, 2,125 muestras de media (máximo 3/4) y
253,125 colisiones por episodio. Se abortó la suite porque ya era imposible
alcanzar el 85 % en sus 50 casos. También se rechazaron actualizaciones por
olvido de escenarios anteriores. Más pasos no equivalen a capacidades nuevas.

El 16 de noviembre se guardó y detuvo esa ejecución a 4.156.256 pasos y se
reanudó desde su checkpoint completo verificado, sin iniciar desde cero.

## Cambio experimental

`configs/experiments/exp-012-observable-navigation.yaml` conserva PPO, cuatro
entornos, normalización heredada congelada y repaso ponderado del escenario 3.
Añade una recompensa auxiliar únicamente en entrenamiento, desde el nivel 6:

`r_aux = gamma * Phi(observación siguiente) - Phi(observación actual)`

Phi vale `8 * max(0, 1 - distancia / RAY_RANGE)` para una muestra objetivo
visible o la base cuando se ha completado la recolección. Sin objetivo visible
vale cero; no consulta el planificador ni distancias a muestras ocultas. En
estados terminales vale cero. En límites temporales conserva el potencial para
el bootstrap. Gamma es 0,995, igual al descuento PPO. La suma descontada de
estas diferencias telescopa: oscilar entre dos posiciones no genera premios
acumulables. La política sigue eligiendo todas las acciones.

Es una hipótesis para facilitar asignación de crédito, no prueba de que el
nivel está superado. No cambia observaciones, mapas, batería, éxito, evaluación
ni recompensas de los niveles 1–5. Las recompensas de entrenamiento del 012
no son directamente comparables con las del 011; comparar evaluaciones.

## Comprobación y límites de almacenamiento

Evaluar cada 16.384 pasos. Aprobar solo con la suite completa de 50 episodios,
al menos 85 % de éxito y máximo 2 choques de media, conservando los niveles
anteriores. Si hay regresión, restaurar los pesos seguros. Cada bloque de
32.768 pasos guarda pesos, optimizador, normalizador, configuración y hashes.

Se conservan dos checkpoints intermedios y un rechazado por versión, además
de pesos finales e hitos. Los replays son compactos y la base SQLite compartida
no se duplica por versión. Las versiones finales y métricas sí siguen creciendo;
no se promete tamaño constante. La campaña comprueba tanto Linux como C: y
solicita guardar/detener al quedar menos de 10 GiB libres.

Control desde Ubuntu, en la carpeta del proyecto:

```sh
.venv/bin/python scripts/training_control.py watch
.venv/bin/python scripts/training_control.py resume
.venv/bin/python scripts/training_control.py stop
```

El supervisor corre separado de la terminal. Cerrar VS Code no le envía una
parada; suspender/apagar el PC o detener WSL sí interrumpe el entrenamiento.
