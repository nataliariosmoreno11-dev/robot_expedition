# Campaña de diez escenarios fijos

Esta campaña amplía el simulador de Natalia. Mantiene PPO, Gymnasium, MuJoCo,
MLflow, DVC y la web Three.js. El ambiente de balizas y sus modelos publicados
conservan su contrato. Los modelos planetarios anteriores necesitan reentrenarse:
la campaña usa `planetary-exploration/2.0.0`, siete acciones y memoria de observaciones.

## Ejecutar y validar

Desde la raíz del repositorio, con las dependencias del README instaladas:

```bash
source .venv/bin/activate
python scripts/campaign.py validate --physical --output logs/campaign-validation.json
python -m pytest -q
python -m uvicorn api.main:app --reload --port 8000
```

En otra terminal:

```bash
cd web
npm ci
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Abre **http://localhost:5173**. Selecciona una misión 01–10 e inicia el escenario.
Las flechas conducen el rover; **Retroceder** es la acción 6 y **Recolectar** la 5.
Reiniciar restaura muestras y geometría y cierra el intento anterior como `reinicio`.
La semilla modifica sensores; nunca escoge otro mapa.

## Mapas y validación física

Las diez definiciones editables están en `configs/campaign/scenario_01.json` a
`scenario_10.json`. Todas usan el mundo existente de 12 × 12 metros: el incremento
de dificultad procede de las decisiones y los rodeos, no de agrandar la superficie.

| Escenario | Problema principal | Muestras |
|---|---|---:|
| 01 | Orientación y regreso | 1 |
| 02 | Barrera con dos rodeos | 2 |
| 03 | Pasillos, intersecciones y desvíos | 3 |
| 04 | Rodear el cráter central protegido por roca | 3 |
| 05 | Corredores, tracción y coste de recorrido | 4 |
| 06 | Cráteres y orden de recolección de dos tipos | 4 |
| 07 | Navegación con polvo, viento y sensores ruidosos | 5 |
| 08 | Barreras alternadas que obligan a alejarse del objetivo | 5 |
| 09 | Más objetivos y una bifurcación adicional | 6 |
| 10 | Ocho objetivos y más bifurcaciones | 8 |

`reference.py` infla los obstáculos por el radio del rover y un margen, descarta
pendientes incompatibles con la tracción y evita cortar esquinas. Dijkstra calcula
costes entre objetivos; una búsqueda exacta por subconjuntos elige su orden.
El coste es distancia ponderada por tracción y pendiente. La energía es una
estimación cinemática conservadora; `--physical` comprueba además la misión en MuJoCo.

La prueba física utiliza un seguidor de puntos exclusivamente para validación.
**No entrena ni controla PPO, y sus éxitos no demuestran aprendizaje.**
Las métricas topológicas se calculan sobre el esqueleto de la rejilla transitable;
dependen de su resolución. `shortest_feasible_path_length` es la longitud del
recorrido de menor **coste de rejilla**, no una garantía de mínimo continuo de distancia.

Las pendientes se limitaron a las que este rover puede transitar en las pruebas.
Gravedad (3,71 m/s²) y densidad atmosférica son aproximaciones científicas; batería,
coeficientes de consumo y `energy_scale` son parámetros de simulación, no mediciones
de una misión real. Los cráteres, dunas y zonas de tracción afectan física y consumo.
El generador procedural anterior sigue disponible para trabajo experimental mediante
`campaign: false`; no se usa en la campaña oficial.

## Entrenamiento y currículo

```bash
python pipelines/train.py --config configs/experiments/exp-007-planetary.yaml --seed 1
```

El presupuesto inicial es 500.000 pasos; no garantiza superar las diez misiones.
El mismo PPO mantiene sus pesos al cambiar de escenario. Recibe las últimas cuatro
observaciones de 33 valores (132 en total) y dispone de siete acciones. Es memoria
temporal limitada, no un mapa global ni un planificador oculto. La política no recibe
coordenadas de muestras ocultas, rutas de referencia ni acciones óptimas.

La recompensa principal es completar y regresar. La cobertura nueva paga una vez
por celda; colisiones, energía, atasco y recolección incorrecta cuestan recompensa.
No se usa distancia a objetivos ocultos para premiar o penalizar movimientos, por
lo que un rodeo no recibe una penalización direccional adicional.

Cada 25.000 pasos se congela la política y su normalización para evaluar el escenario
actual. Se exige inicialmente ≥85 % de éxito en ≥50 semillas distintas, ≤2 colisiones
medias, inventario completo y batería positiva en los éxitos. Los umbrales están en
`environment.curriculum_settings`. Las estadísticas de entrenamiento no promueven.
Un 20 % de los episodios repasa escenarios anteriores. Antes de avanzar se evalúa
su retención; un fallo devuelve el currículo al escenario perdido.

Cada promoción guarda pesos, optimizador, normalización, contador y estado curricular
en `models/<run_id>/mastered-scenario-XX-<paso>/`. El estado activo pertenece al run,
no se comparte accidentalmente entre experimentos distintos. Una interrupción normal
guarda también el checkpoint final y cierra los episodios abiertos.

Para continuar, copia el YAML y establece:

```yaml
parent_run_id: local:/ruta/absoluta/models/<run_id>
```

También admite un `parent_run_id` de MLflow o la carpeta de un checkpoint de promoción.
El presupuesto de la siguiente ejecución se añade al contador anterior. La base
SQLite anterior se conserva.

Prueba corta del pipeline (verifica funcionamiento, **no aprendizaje**):

```bash
python pipelines/train.py --config configs/experiments/exp-007-planetary.yaml \
  --seed 1 --budget-timesteps 128 --eval-episodes 2 --max-episode-steps 8 \
  --n-steps 8 --batch-size 16
```

## Dominio, robustez y evidencia de aprendizaje

```bash
python scripts/campaign.py evaluate --scenario scenario_08 --kind mastery \
  --model models/<run_id>/ppo_model.zip --vecnormalize models/<run_id>/vecnormalize.pkl \
  --episodes 50 --output logs/dominio-08.json
python scripts/campaign.py evaluate --scenario scenario_08 --kind robustness \
  --model models/<run_id>/ppo_model.zip --vecnormalize models/<run_id>/vecnormalize.pkl \
  --episodes 50 --output logs/robustez-08.json
```

Dominio usa el inicio original. Robustez perturba posición y rumbo de inicio sin
cambiar el mapa ni promover el currículo. Las evaluaciones externas se guardan en
`logs/campaign_evaluations.sqlite`; para visualizarlas arranca la API con
`EXPEDITION_EPISODE_DB=logs/campaign_evaluations.sqlite`. La campaña no afirma
generalización a mapas inéditos: las pruebas de mapas reservados no se implementaron.

Busca mejoras en éxito **evaluado**, colisiones, energía y eficiencia, comparando
la misma misión y las mismas semillas. La web separa entrenamiento, dominio,
robustez y manual. Una subida de recompensa sola no demuestra dominio.
Los runs cortos y el seguidor de referencia no se presentan como políticas exitosas.

## Registro, reproducción y comparación

SQLite/WAL guarda cada paso incrementalmente, comprimido, con observación, pose,
acción, recompensa, inventario, recursos y versión de política. Todos los intentos
de campaña se conservan completos, incluidos fallos y reinicios. Un cierre abrupto
puede dejar un intento `en_progreso`, pero sus pasos confirmados siguen reproducibles.
La política actual no interviene al reproducir. No se recuperan los replays que la
versión anterior ya hubiera podado antes de esta actualización.

```bash
python scripts/campaign.py show scenario_08
python scripts/episodes.py list --limit 20 --offset 0
python scripts/episodes.py show 3427
python scripts/episodes.py replay 3427 --output logs/replay-3427.json
python scripts/episodes.py history scenario_08 --offset 0 --limit 100
python scripts/episodes.py compare 100 500 1000
python scripts/episodes.py attempts scenario_08 1 100 500
python scripts/episodes.py progress scenario_08 --mode entrenamiento
```

En la web selecciona el escenario, filtra el tipo de intento y pulsa **Actualizar**.
Puedes escribir cualquier número histórico, cargar su replay, pausar, reiniciar la
reproducción, cambiar velocidad y mostrar u ocultar trayectoria y referencia. La
referencia solo se muestra en replays. **Comparar intentos** presenta sus métricas;
**Primeros, intermedios, recientes y mejor** consulta todo el historial mediante SQL.
Los gráficos muestran los últimos 100 intentos del filtro, no todo el histórico.

La eficiencia solo se calcula en misiones completadas con costes comparables:
referencia de menor coste sobre las mismas posiciones reales de interacción e inicio/
fin, dividida por el coste del rover. Si queda fuera de la rejilla conservadora se
muestra sin valor. La discretización puede producir cocientes ligeramente mayores
que 1; no se recortan ni se confunden con distancia euclidiana.

## Verificaciones y límites

La evidencia ejecutada está en `docs/evidencia/campaign-validation.json` y
`docs/evidencia/campaign-checks.json`. Las pruebas cubren mapas fijos, contactos,
recolección, memoria, ausencia de recompensa privilegiada, currículo, persistencia,
API, PPO real y continuación del optimizador. El contrato está en
`docs/contratos-planetario.md`.

La validación física demuestra una solución por mapa bajo la semilla documentada;
no es una prueba formal para todo ruido posible. La dificultad es multidimensional:
no todas las métricas crecen estrictamente de un escenario al siguiente. El escenario
10 añade decisiones y objetivos respecto al 9, conservando el mismo tamaño de mapa.
