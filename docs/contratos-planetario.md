# Contrato del simulador planetario

El ambiente nuevo se activa de forma explícita con `PlanetaryExplorationEnv` o
`environment=planetary`. `BeaconExpeditionEnv`, sus cinco acciones, sus veinte
observaciones, los modelos publicados y las rutas de evaluación anteriores se
mantienen sin cambios.

## Reproducibilidad del escenario

Un escenario queda identificado por `(generator_version, planet, difficulty_level,
seed)`. Con esos cuatro valores, el generador produce exactamente el mismo terreno,
clima, rocas, cráteres, dunas, tipos y posiciones de muestras. Antes de aceptar un
escenario comprueba rutas transitables, pendientes, tracción, capacidad de batería,
inventario y límite de pasos. Si un intento no es válido, incrementa un contador de
intento determinista y vuelve a generarlo.

El perfil físico `mars-v1` declara gravedad de `3.71 m/s²`, densidad atmosférica
aproximada de `0.020 kg/m³`, temperatura de referencia de `-55 °C`, resistencia a la
rodadura, fricción y parámetros aerodinámicos del rover. Son aproximaciones de
simulación documentadas, no mediciones de una misión real.

## Acciones

| Índice | Acción |
|---:|---|
| 0 | avanzar |
| 1 | avanzar + izquierda |
| 2 | avanzar + derecha |
| 3 | girar izquierda |
| 4 | girar derecha |
| 5 | recolectar |

`recolectar` solo funciona dentro del radio de colección. Una muestra del tipo
incorrecto, una repetida, un inventario lleno o intentar recolectar sin una muestra
cercana tienen penalizaciones distintas. La misión solo termina con éxito después
de reunir el número solicitado del tipo correcto y regresar a la base.

## Observación de 33 valores

| Índices | Contenido | Escala |
|---|---|---|
| 0–8 | nueve rayos de distancia | metros, limitados por visibilidad |
| 9 | distancia a objetivo visible (o alcance máximo) | metros |
| 10–11 | rumbo relativo visible; ambos cero si es desconocido | `[-1, 1]` |
| 12–13 | velocidad y giro | normalizados |
| 14 | batería | `[0, 1]` |
| 15–17 | fracción recogida, restante e inventario | `[0, 1]` |
| 18–20 | tipo solicitado: roca, hielo, mineral | one-hot |
| 21–22 | muestra visible y distancia | indicador, metros |
| 23–25 | tipo de muestra visible | one-hot |
| 26–29 | gravedad, tracción, visibilidad y viento | normalizados |
| 30–31 | inclinación longitudinal y lateral | radianes |
| 32 | dificultad curricular | nivel / 8 |

La especificación del manifiesto guarda tamaño de observación y acciones. Un modelo
del contrato de balizas no puede servirse dentro de una sesión planetaria.
Las posiciones de muestras ocultas no forman parte de la observación. El detector
solo informa una muestra dentro del alcance afectado por polvo y con línea de visión
sin rocas; después de completar el inventario, la base sí se usa como destino conocido.

## Persistencia de episodios

`EpisodeStore` usa SQLite con WAL. Crea un registro al iniciar y confirma cada paso
de forma incremental; no conserva el episodio completo en memoria. Guarda todos los
intentos, incluidos fallos, reinicios e interrupciones:

- identidad de run, versión de política, semilla, escenario, planeta y dificultad;
- mapa y escenario completos comprimidos;
- pose, inventario, estado de cada muestra y ambiente en cada paso;
- acción, recompensa instantánea y acumulada, evento, choque y señales de fin;
- resultado, causa, duración, recorrido y muestras recogidas.

`replay_episode(n)` devuelve los cuadros guardados. Nunca vuelve a ejecutar la
política, por lo que cambiar un checkpoint posteriormente no altera un episodio.
El esquema actual es `1`; una versión incompatible se rechaza en lugar de reinterpretar
datos antiguos.

## API planetaria

```text
POST /sim/reset              environment=planetary, difficulty_level=1..8
POST /sim/step               action=0..5
GET  /episodes
GET  /episodes/{n}
GET  /episodes/{n}/replay
GET  /episodes/{n}/scenario
POST /episodes/compare       episode_numbers=[...]
```

La ubicación se configura con `EXPEDITION_EPISODE_DB`; por defecto es
`logs/planetary_episodes.sqlite`.
