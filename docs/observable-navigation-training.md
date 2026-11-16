# Experimento 012: navegaci?n observable

## Motivo y punto de partida

El experimento 011 sigui? sin aprobar el escenario 6. En la evaluaci?n de
4.137.824 pasos registr? 0/8 ?xitos, 2,125 muestras de media (m?ximo 3/4) y
253,125 colisiones por episodio. Se abort? la suite porque ya era imposible
alcanzar el 85 % en sus 50 casos. Tambi?n se rechazaron actualizaciones por
olvido de escenarios anteriores. M?s pasos no equivalen a capacidades nuevas.

El 16 de noviembre se guard? y detuvo esa ejecuci?n a 4.156.256 pasos y se
reanud? desde su checkpoint completo verificado, sin iniciar desde cero.

## Cambio experimental

`configs/experiments/exp-012-observable-navigation.yaml` conserva PPO, cuatro
entornos, normalizaci?n heredada congelada y repaso ponderado del escenario 3.
A?ade una recompensa auxiliar ?nicamente en entrenamiento, desde el nivel 6:

`r_aux = gamma * Phi(observaci?n siguiente) - Phi(observaci?n actual)`

Phi vale `8 * max(0, 1 - distancia / RAY_RANGE)` para una muestra objetivo
visible o la base cuando se ha completado la recolecci?n. Sin objetivo visible
vale cero; no consulta el planificador ni distancias a muestras ocultas. En
estados terminales vale cero. En l?mites temporales conserva el potencial para
el bootstrap. Gamma es 0,995, igual al descuento PPO. La suma descontada de
estas diferencias telescopa: oscilar entre dos posiciones no genera premios
acumulables. La pol?tica sigue eligiendo todas las acciones.

Es una hip?tesis para facilitar asignaci?n de cr?dito, no prueba de que el
nivel est? superado. No cambia observaciones, mapas, bater?a, ?xito, evaluaci?n
ni recompensas de los niveles 1?5. Las recompensas de entrenamiento del 012
no son directamente comparables con las del 011; comparar evaluaciones.

## Comprobaci?n y l?mites de almacenamiento

Evaluar cada 16.384 pasos. Aprobar solo con la suite completa de 50 episodios,
al menos 85 % de ?xito y m?ximo 2 choques de media, conservando los niveles
anteriores. Si hay regresi?n, restaurar los pesos seguros. Cada bloque de
32.768 pasos guarda pesos, optimizador, normalizador, configuraci?n y hashes.

Se conservan dos checkpoints intermedios y un rechazado por versi?n, adem?s
de pesos finales e hitos. Los replays son compactos y la base SQLite compartida
no se duplica por versi?n. Las versiones finales y m?tricas s? siguen creciendo;
no se promete tama?o constante. La campa?a comprueba tanto Linux como C: y
solicita guardar/detener al quedar menos de 10 GiB libres.

Control desde Ubuntu, en la carpeta del proyecto:

```sh
.venv/bin/python scripts/training_control.py watch
.venv/bin/python scripts/training_control.py resume
.venv/bin/python scripts/training_control.py stop
```

El supervisor corre separado de la terminal. Cerrar VS Code no le env?a una
parada; suspender/apagar el PC o detener WSL s? interrumpe el entrenamiento.
