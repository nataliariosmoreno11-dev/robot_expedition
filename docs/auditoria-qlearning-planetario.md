# Auditoría del Q-learning planetario (niveles 1-8)

**Fecha:** 2026-11-19 · **Responsable:** Nicol · **Alcance:** política `planetary-rule`
(guía), política `planetary-qlearning` y sesiones de la API.

## Síntoma

En producción el Q-learning resolvía solo el nivel 1 (~70 %). En 2.400 episodios de
entrenamiento no completó **ninguna** misión de los niveles 2 a 8. Entrenar más no
cambiaba el resultado.

## Causas encontradas

| # | Causa | Evidencia |
|---|---|---|
| 1 | **El guía no sabía resolver los niveles.** El Q-learning lo sigue el 95 % de las veces durante la exploración. Su "arco suave" de búsqueda eran 12 pasos de `avanzar + izq`, pero esa acción gira 0,8 rad/s × 0,5 s = 23° por paso: el arco era un giro de 270° que llevaba al rover contra la pared oeste. La maniobra de escape lo devolvía a la misma pared. | Con todos los pasos de diseño: atascado el 95 % del episodio en L2, L3, L4, L6 y L8; 0 de 24 éxitos fuera de L1. |
| 2 | **Límite de pasos imposible.** La API cortaba el Q-learning en 50 pasos (L1-L4) o 100 (L5-L8). Los escenarios están diseñados para 650-1.700 pasos y una ruta eficiente necesita 90-450. | L2 exige recorrer unos 20 m a ~0,2 m/paso. |
| 3 | **Corte por recompensa mínima (−20).** La penalización por paso (0,04) más el consumo de batería llegan a −20 en unos 300 pasos, aunque la misión vaya bien. | Fin `limite_recompensa` en la mayoría de episodios de L2-L8. |
| 4 | **Corte "sin progreso" a los 35 pasos.** El regreso a la base cruza celdas ya exploradas: más de 35 pasos sin descubrir celdas cortaba la misión justo al volver. | — |
| 5 | **Sensores atrasados 1,5 s.** La API apila 4 observaciones (la actual al final), pero el guía y `discretize_planetary` leían `observation[:33]`, la de hace 3 pasos. | Velocidad leída en 0 durante 4 pasos mientras el rover ya avanzaba. |
| 6 | **Peligros invisibles.** La grieta fija en (3,5; 3,5) es un hueco y los cráteres son depresiones; el LIDAR son rayos horizontales a 0,3 m que no detectan el suelo. | Vuelco en L5 y caída en L7 en el mismo punto (3,4; 3,3). |

## Cambios

1. **Guía planificador** ([`robot_expedition/planetary/planner.py`](../robot_expedition/planetary/planner.py)),
   que reemplaza a la política reactiva:
   - Recibe solo el **mapa de misión estático**: área, base, rocas, grieta, cráteres,
     dunas y zonas de baja tracción, más la pose de aterrizaje. **Nunca recibe la
     posición de las muestras**; las descubre con la cámara de la observación pública
     y estima dónde están con el rumbo y la distancia observados (hay un test que lo
     verifica).
   - Se ubica por **odometría** con la velocidad y el giro observados, y la corrige
     comparando los 9 rayos del LIDAR con los que produciría cada pose candidata contra
     rocas y paredes conocidas (error del modelo de rayos: 3-5 cm).
   - Recorre puntos de observación que cubren el área con la cámara y traza rutas con
     **A\*** con márgenes de seguridad. La grieta y el núcleo letal de cada cráter
     (donde el rover bajaría más de 0,15 m) se bloquean; los bordes transitables solo
     cuestan más.
   - Mantiene la meta elegida hasta alcanzarla (antes oscilaba entre dos puntos
     equidistantes) y la elige por longitud de ruta, no en línea recta.
2. **Q-learning sobre el guía:** el estado incluye la acción que sugiere el guía
   (`STATE_SIZE` 11 → 12), así la tabla aprende en qué contexto seguirla o corregirla.
   La actualización Q se completa un paso después, cuando existe la sugerencia para
   el estado siguiente. Junto a la grieta o un cráter, un escudo de seguridad
   ejecuta siempre la acción del guía (`shielded_decisions` en el estado).
3. **Sesiones de la API:** límite de pasos del escenario (la batería es el límite
   real), sin piso de recompensa y "sin progreso" a los 120 pasos.
4. **Sensores actuales:** el guía y `discretize_planetary` usan la última de las
   observaciones apiladas.
5. **Checkpoints anteriores:** un checkpoint de 11 componentes no se borra. Se
   conserva renombrado como `*-estado-anterior.npz`, aparece en la lista de
   incompatibles y la API arranca con el checkpoint base `qlearning-seed-v2`.

## Discretización

El ambiente es continuo: la observación tiene 33 componentes reales (132 con el
historial). Q-learning tabular necesita estados discretos, así que
`discretize_planetary` los reduce a 12 enteros. La discretización usa **solo la
lectura más reciente**; el historial apilado se descarta.

| # | Componente | Origen | Cortes (intervalos) |
|---|---|---|---|
| 1 | Distancia libre al frente | mínimo de los rayos 4-6 | 0,45 · 0,8 · 1,5 · 3,0 m (5) |
| 2 | Espacio a la izquierda | media de los rayos 1-4 | 0,7 · 1,5 · 3,0 m (4) |
| 3 | Espacio a la derecha | media de los rayos 6-9 | 0,7 · 1,5 · 3,0 m (4) |
| 4 | Rumbo al objetivo | `atan2(sin, cos)` | ±0,22 · ±0,8 · ±2,35 rad (7) |
| 5 | Distancia al objetivo | `navigation_distance` | 0,62 · 1,4 · 3,0 · 5,0 m (5) |
| 6 | Batería | fracción | 0,2 · 0,45 · 0,7 (4) |
| 7 | Avance de la misión | fracción recolectada | 0,01 · 0,5 · 0,99 (4) |
| 8 | ¿Muestra visible? | booleano | 2 |
| 9 | ¿Es del tipo pedido? | one-hot | 2 |
| 10 | Tracción local | coeficiente | 0,55 · 0,72 · 0,9 (4) |
| 11 | Nivel | fracción | 0,25 · 0,5 · 0,75 (4) |
| 12 | Acción sugerida por el guía | 0-6 o "sin guía" | 8 |

**Qué se pierde:**

- **Posición y memoria.** Ningún componente dice dónde está el rover ni qué zonas
  ya exploró. Dos situaciones con los mismos sensores pero en partes distintas del
  mapa son el mismo estado.
- **Detalle del LIDAR.** 9 distancias se resumen en 3 intervalos (frente,
  izquierda, derecha): se pierde la forma del obstáculo.
- **Velocidad, giro, viento, inclinación y tipo de muestra** no entran en el estado.

**Cuánto se pierde (medido).** En 24 misiones del guía planificador (L1-L8, 3
semillas, 5.450 pasos), cada observación continua fue distinta, pero cayeron en
1.073 estados de 11 componentes, unas 5 observaciones por estado. En 458 de esos
estados el guía tomó más de una acción distinta. **En el 32 % de los pasos, la
acción correcta difería de la acción mayoritaria de su estado**: la tabla v1 no
podía representar esa decisión con ningún valor Q. Al agregar la acción del guía
(componente 12), ese aliasing desaparece por construcción: la planificación que
requiere posición y memoria la resuelve el guía, y la tabla aprende cuándo
seguirlo. Lo sigue o corrige según el contexto sensorial, en 1.770 estados para
esas mismas misiones.

## Resultados

Con los límites de diseño de cada escenario y semillas distintas de las de
entrenamiento:

| Política | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | Total |
|---|---|---|---|---|---|---|---|---|---|
| Guía anterior | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/24 |
| Guía planificador | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 | **79/80** |
| Q-learning `seed-v1` (50/100 pasos) | ~7/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| Q-learning `seed-v2` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 7/10 | 10/10 | 8/10 | **75/80** |

El planificador completa las misiones con 0-2 choques y le queda entre 35 % y 95 %
de batería. El Q-learning `seed-v2` se entrenó con 480 episodios (12 rondas por los 8
niveles; 90 % de éxito durante el entrenamiento) y se evaluó con semillas distintas
(6000-6009) sobre una copia del checkpoint. Su archivo SHA-256 es
`1e203e0d39cde8c5f4ca2291af5e9cbcf8b5b0bb0d92ad96e2c0d4d64f99ac5c`.

**Escudo de seguridad:** junto a la grieta o un cráter, el Q-learning siempre ejecuta
la acción del guía. Sin escudo hubo 5 caídas en 80 episodios (73/80); con escudo,
ninguna. Los 5 fallos restantes son cortes `sin_progreso` (120 pasos sin explorar
celdas nuevas) cuando las acciones aprendidas forman un ciclo en L6 y L8.

Evidencia completa por nivel (pasos, batería, choques y causa de fin):
[evidencia/evaluacion-planificador-qlearning.json](evidencia/evaluacion-planificador-qlearning.json).

## Cómo reproducir

```bash
python scripts/train_planetary_qlearning.py --out planetary_qlearning.npz --rounds 12 --seeds-per-level 5
python scripts/evaluate_planetary_policies.py --episodes 10 --q-checkpoint planetary_qlearning.npz
```

## Límites conocidos

- El guía usa el mapa estático de la misión, el equivalente a imágenes orbitales.
  Para que el rover *perciba* la grieta habría que cambiar la observación, y eso
  invalidaría los modelos PPO de la campaña. Se decidió no hacerlo.
- Sin disco persistente en Render, lo aprendido en vivo se pierde en cada reinicio:
  la API vuelve al checkpoint base.
