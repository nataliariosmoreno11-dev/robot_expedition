# Ambiente v1.1 · Robot Expedition

> Ficha de entrega del ambiente (sección 02 del enunciado). Reemplaza a la
> [ficha v0.1](ambiente-v0.1.md), que se conserva como historial del diseño.
>
> - **Versión:** `robot-expedition-3d/1.1.0` (`ENVIRONMENT_VERSION`).
> - **Código:** [`robot_expedition/envs/beacon_expedition.py`](../robot_expedition/envs/beacon_expedition.py).
> - **Niveles:** [`configs/levels/`](../configs/levels).
> - **Contrato de interfaz:** [`docs/contratos.md`](contratos.md).
>
> Cada número de esta ficha sale de esas constantes o de una ejecución
> registrada. Los mapas se generan desde el código con
> `python scripts/draw_level_maps.py`.

## 1. Misión

Un rover con tracción diferencial:
1. sale de la base;
2. visita las balizas **en orden** (N1: B1; N2 y N3: B1 y luego B2);
3. vuelve a la base antes de agotar la batería.

Además, no debe caer en la grieta, volcar ni chocar.

El mundo es 3D y la física la calcula **MuJoCo 3.3.6**:
- gravedad de Marte, 3.71 m/s²;
- terreno con relieve (heightfield de 121 × 121 puntos, uno cada 0.1 m);
- una grieta real de 0.5 m de profundidad;
- rocas y paredes con colisión;
- arena que frena al rover.

## 2. Unidades y rangos

| Magnitud | Unidad | Rango | Constante |
|---|---|---|---|
| Mapa | m | x ∈ [0, 12], y ∈ [−6, 6] | `MAP_W`, `MAP_H` |
| Base | m | centro (1.5, 0), cuadrado de 2 × 2 | `BASE` |
| Balizas | m | B1 (9.5, 3.5), B2 (9.5, −3.5), radio de visita 0.6 | `BEACONS`, `VISIT_RADIUS` |
| Grieta | m | centro (3.5, 3.5), 1.0 × 0.4, 0.5 de profundidad | `CRACK`, `CRACK_DEPTH` |
| Arena | m | centro (7, −4.5), 2.0 × 1.4; arrastre `−12·v` N | `SAND` |
| Velocidad del rover | m/s | pedida de 0 a 0.6 | `V_MAX` |
| Giro | rad/s | pedido de −1.2 a 1.2 | `W_MAX` |
| Torque por rueda | N·m | ±1.5 | `forcerange` |
| Batería | % | [0, 100] | — |
| Rayos | m | [0, 5], 9 rayos de −90° a +90° cada 22.5° | `N_RAYS`, `RAY_RANGE` |
| Tiempo | pasos | [0, 400]; 1 paso = 0.5 s simulados | `max_steps` |

## 3. Observación

La observación es un vector `float32` de 20 valores, `Box(20)`. El agente
**no** ve `x, y`.

| Índice | Contenido | Unidad | Rango |
|---|---|---|---|
| 0–8 | 9 rayos de distancia (derecha → izquierda) con ruido N(0, σ) | m | [0, 5] |
| 9 | Distancia al objetivo actual | m | [0, 17] |
| 10–11 | sen y cos del ángulo al objetivo, relativo al frente | — | [−1, 1] |
| 12 | Velocidad hacia adelante / 0.6 | — | [−1, 1] |
| 13 | Velocidad de giro / 1.2 | — | [−1, 1] |
| 14 | Batería / 100 | — | [0, 1] |
| 15–17 | Etapa en one-hot: ir a B1, ir a B2, volver a la base | — | {0, 1} |
| 18–19 | Cabeceo y alabeo del chasis | rad | [−π, π] |

**La etapa 3 siempre significa "volver a la base"**, en todos los niveles. En
N1, al visitar B1 la etapa salta de 1 a 3. Así el índice 17 tiene el mismo
significado en N1, N2 y N3, y un modelo entrenado en un nivel puede evaluarse
en otro.

Los rayos se lanzan en 3D con `mujoco.mj_ray`. Detectan paredes, rocas y el
bloqueo. **No detectan la grieta** (es un hueco) ni el terreno.

## 4. Acciones, paso físico y frecuencia de decisión

La acción es `Discrete(5)`. Cada una pide una velocidad y un giro, y el
ambiente los convierte en la velocidad de cada rueda:
`ω_rueda = (v ∓ ω · 0.21) / 0.1`.

| # | Acción | v pedida (m/s) | ω pedida (rad/s) |
|---|---|---|---|
| 0 | avanzar | 0.6 | 0 |
| 1 | avanzar + izq | 0.4 | +0.8 |
| 2 | avanzar + der | 0.4 | −0.8 |
| 3 | girar izq | 0 | +1.2 |
| 4 | girar der | 0 | −1.2 |

Cualquier otro valor es inválido. Gymnasium lo rechaza y la API responde 422.

**Tiempo:**
- **Paso físico:** `TIMESTEP = 0.02 s` (50 Hz). MuJoCo integra la gravedad, los
  motores y los contactos.
- **Paso de decisión:** `DT = 0.5 s` (**2 Hz**). Cada `step(a)` integra 25 pasos
  físicos en 5 bloques de 0.1 s. Entre bloques se revisan los choques y la arena.
- La acción se mantiene durante los 0.5 s. El agente no puede corregir dentro
  de un paso.

## 5. Recompensa

```
r_t = (d_{t-1} − d_t) − 0.01 − 0.02·E_t − 0.5·choque_t
      + 10                 al visitar la baliza de la etapa actual (una vez)
      + 50 + 20·B/100      al llegar a la base en la etapa 3 (éxito)
      − 50                 al caer en la grieta, volcar o agotar la batería
```

con `E = min(1, (v/0.6)² + 0.25·(ω/1.2)²)` medido sobre la velocidad real.
La batería baja `0.1 + 0.4·E` por paso.

El retorno **no** es la métrica de la misión. Solo se compara dentro de un
mismo nivel; la métrica principal es la tasa de éxito.

## 6. Éxito, fallos y truncamiento

| Fin | Condición | Gymnasium | `info["end"]` |
|---|---|---|---|
| Éxito | Todas las balizas del nivel visitadas en orden, rover dentro de la base y `B > 0` | `terminated` | `exito` |
| Grieta | Chasis sobre la grieta o `z < −0.15 m` | `terminated` | `grieta` |
| Vuelco | Eje vertical del chasis inclinado más de ~73° (`upright < 0.3`) | `terminated` | `volco` |
| Batería | `B = 0` | `terminated` | `bateria` |
| Tiempo | 400 pasos (200 s simulados) sin ninguno de los anteriores | `truncated` | `tiempo` |

`terminated` y `truncated` son distintos a propósito. Cuando se acaba el
tiempo, el estado no es terminal y el valor futuro se sigue estimando
(bootstrapping). Las pruebas de estas reglas están en
`tests/test_beacon_expedition.py` y `tests/test_levels.py`.

## 7. Niveles

Los tres niveles comparten el mismo código, la misma observación y las mismas
acciones. Solo cambia el archivo YAML del nivel, y cada uno tiene su propio
`environment_hash` (código del ambiente + nivel resuelto).

| | N1 | N2 | N3 |
|---|---|---|---|
| Archivo | [n1.yaml](../configs/levels/n1.yaml) | [n2.yaml](../configs/levels/n2.yaml) | [n3.yaml](../configs/levels/n3.yaml) |
| Balizas | 1 (B1) | 2 (B1 → B2) | 2 (B1 → B2) |
| Rocas activas | 2 (R1, R2) | 5 | 5 |
| Dunas | no (terreno plano salvo la grieta) | 6 dunas de 0.12–0.30 m | igual que N2 |
| Bloqueo del corredor | no | no | **sí**: aparece al visitar B1 en (9.7, 0), 2.2 × 0.6 × 0.8 m |
| Ruido de rayos σ | 0.02 m | 0.02 m | **0.08 m** |
| Rayo muerto | — | — | **rayo 4 (frontal, 0°)**: siempre marca 5 m |
| Ruta de referencia (centro a centro) | 17.46 m | 24.46 m | 24.46 m |
| Suite fija | `suite-n1-v1` (semillas 100000–100049) | `suite-n2-v1` (200000–200049) | `suite-n3-v1` (300000–300049) |

### N1 · Misión base

![Mapa N1](mapas/mapa-n1.svg)

- **Decisión nueva:** orientarse hacia un objetivo que no ve y rodear un
  obstáculo. La línea recta base → B1 cruza R1, y la grieta queda cerca del
  rodeo por el norte.
- **Dificultad:** se mide con la tasa de éxito y la eficiencia de ruta
  (17.46 m / distancia recorrida).
- **Referencias:** la política aleatoria tiene 0 % de éxito y la de regla
  100 % ([resultados](resultados/baselines.md)).

### N2 · Misión completa

![Mapa N2](mapas/mapa-n2.svg)

- **Decisiones nuevas:**
  - **Secuenciar dos objetivos.** La etapa cambia el objetivo, y entrar al radio
    de B2 antes que al de B1 no cuenta.
  - **Administrar la batería** en una ruta de ~24.5 m.
  - **Elegir entre cruzar dunas o rodearlas.** Las dunas inclinan el rover,
    frenan y aumentan el riesgo de volcar.
- **Dificultad:** se mide con el éxito, las balizas visitadas en media, la
  batería final de los éxitos y los vuelcos.

### N3 · Incidente: bloqueo y sensor degradado

![Mapa N3](mapas/mapa-n3.svg)

- **Decisión nueva:** **replanificar a mitad de la misión.** Al visitar B1
  aparece una roca que corta el camino directo B1 → B2. Hay que rodearla por el
  pasillo junto a la pared este o por el oeste de R3, con un sensor frontal
  muerto (rayo 4) y el resto de los rayos más ruidosos (σ = 0.08 m).
- **Dificultad:** se mide con el éxito, los choques por episodio (empujar contra
  el bloqueo) y los fallos por tiempo.
- **Uso:** este es el escenario del incidente controlado ([incidente](incidente.md)).
  Un modelo entrenado en N2 nunca vio el bloqueo.

## 8. Escenarios variables y simplificaciones

- **Por defecto, las balizas y rocas siguen fijas** para reproducir los modelos
  y resultados v1.0. El experimento `exp-006` activa `varied-v1`: seis
  escenarios cambian B1/B2, la cantidad y posición de rocas y el bloqueo según
  la semilla. La suite `suite-n2-generalization-v1` los cubre de forma
  equilibrada. Ver [escenarios-variados.md](escenarios-variados.md).
- La base, la grieta, la arena y las dunas conservan su posición en v1.1.
- **El rover es un cuerpo rígido.** Tiene dos ruedas esféricas con motor de
  velocidad y dos apoyos sin fricción. No hay suspensión, deslizamiento de
  orugas ni daño.
- **La arena es una fuerza de arrastre** `−12·v` aplicada al chasis. No es un
  modelo de suelo granular.
- **La batería** es un modelo lineal de consumo según la velocidad real. No
  hay regeneración ni temperatura.
- **Los rayos son ideales** salvo el ruido gaussiano y el rayo muerto: no hay
  oclusión parcial ni reflejos.
- **Las decisiones van a 2 Hz.** Un sistema real decidiría más rápido; aquí
  se eligió 2 Hz para que un episodio completo cueste ~0.6 ms por paso y se
  pueda entrenar en CPU.

## 9. Pérdida de resolución en la discretización del Q-learning

El Q-learning tabular ([`robot_expedition/qlearning.py`](../robot_expedition/qlearning.py))
no usa los 20 floats. Los convierte en un estado discreto de
3 × 4 × 8 × 3 × 3 × 3 = **2 592 estados**:

| Información | En la observación | En el estado discreto | Qué se pierde |
|---|---|---|---|
| Distancia al objetivo | continua de 0 a 17 m | 4 franjas: < 1, < 2.5, < 5 y ≥ 5 m | Más allá de 5 m todas las distancias son iguales |
| Ángulo al objetivo | continuo | 8 sectores de 45° | Errores de rumbo de hasta ±22.5° no se distinguen |
| Rayos | 9 distancias | Mínimo de 3 grupos (derecha, frente, izquierda) en 3 franjas | Qué rayo exacto ve el obstáculo y a qué distancia exacta |
| Velocidad, giro y batería | 3 valores | no se usan | El agente no sabe si está frenado ni cuánta batería le queda |
| Cabeceo y alabeo | 2 valores | no se usan | No puede anticipar un vuelco en una duna |

Consecuencias que se ven en los resultados:
- La grieta no se ve con los rayos, así que el Q-learning solo puede evitarla
  por el ángulo y la distancia al objetivo. En N1 cae en la grieta en varios
  episodios ([resultados](resultados/baselines.md)).
- PPO usa la observación completa, normalizada con `VecNormalize`, así que no
  tiene estas pérdidas.
- Con el rayo frontal muerto (N3), el grupo "frente" del Q-learning solo
  depende de los rayos a ±22.5°.

## 10. La velocidad de animación no cambia las reglas

La simulación siempre corre en la API (o en el CLI) con `DT = 0.5 s` y
`TIMESTEP = 0.02 s`, y devuelve los cuadros ya calculados. El selector de
velocidad de la web (`#speedSelect`) solo cambia cada cuánto se dibuja el
siguiente cuadro. No cambia el número de pasos, la física, la recompensa ni el
resultado. Si se evalúa el mismo modelo con la misma semilla, se obtienen los
mismos cuadros a cualquier velocidad de reproducción
(`tests/test_api.py::test_evaluate_matches_the_offline_suite`).

## 11. Cambios desde v0.1

| v0.1 | v1.0 |
|---|---|
| Un solo escenario y flags de CLI (`--perturbation`, `--ray-noise`) | Tres niveles versionados en YAML, con `environment_hash` por nivel |
| La etapa one-hot dependía del número de balizas | La etapa 3 siempre es "volver a la base" |
| Evaluación ad hoc de 50 semillas | Suites fijas por nivel con SHA-256 y un test reservado fuera de Git ([contratos](contratos.md)) |
| Solo Q-learning | PPO (Stable-Baselines3) y Q-learning, registrados en MLflow con manifiesto |
| Bípedo como segundo ambiente | El bípedo queda fuera de la entrega; el contrato de evaluación es el del rover |
