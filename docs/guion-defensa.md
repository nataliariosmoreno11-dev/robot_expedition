# Guion de la defensa · Robot Expedition

Sigue el guion sugerido del enunciado (sección 18). La duración se ajusta al
tiempo asignado; cada bloque nombra a quien lo presenta, pero **todos** deben
poder responder las preguntas de la sección final.

**Antes de empezar:**
1. Abrir `https://robot-expedition-api.onrender.com/health`. Si el servicio
   está dormido (plan gratuito de Render), despierta en cerca de 1 min.
2. Abrir la web en Vercel y comprobar que la insignia dice *API disponible*.
3. Comprobar en `/version` que `release` y `git_commit` son los del correo de entrega.

## Presentación final v1.3 · 10 minutos (lineamientos de Teams)

| Min | Bloque (lineamiento) | Quién | Qué mostrar |
|---|---|---|---|
| 0-2 | 1. Problema y contexto de negocio | Suleily | Exploración autónoma: un rover debe traer muestras sin teleoperación (retardo de comunicación) ni gastar de más la batería. Valor: misiones más cortas y seguras. Fase 1: ir a balizas; fase 2: recolectar muestras |
| 2-4 | 2. Modelamiento del ambiente | Natalia | Contrato planetario: 132 observaciones, 7 acciones, 0,5 s por decisión; éxito, fallos y truncamiento ([informe §8](informe.md)); los 8 escenarios |
| 4-5 | 3. Recompensas y penalizaciones | Natalia | Qué premia y castiga. Verificación: el retorno no equivale al éxito, porque un candidato con más retorno puede no completar la misión ([experimentos](experimentos.md)) |
| 5-7 | 4. Modelo y experimentación | Tomás y Nicol | Baselines (aleatorio 0 %, regla). PPO de campaña: exp-008 a exp-013 no aprobaron L6 (empeoró o se estancó); exp-014/015 aprobaron L1-L8. Q-learning: v1 solo L1 → diagnóstico → v2 75/80. Fase 1: finalistas con 3 semillas |
| 7-8 | 5. Versionamiento y trazabilidad | Nicol | Tag `v1.3` y SHA en `/version`; manifiesto de un modelo (`manifests/planetary/`); recuperar una versión anterior (rollback del catálogo y `qlearning-seed-v1` conservado) |
| 8-10 | 6. Demostración y conclusiones | Natalia y Suleily | Web: un modelo histórico y el recomendado; Q-learning en L5-L7; limitaciones (PPO memoriza mapas, guía con mapa estático, Render sin disco) y siguiente hipótesis (PPO procedural) |

### Preguntas probables sobre la fase 2

- **¿Por qué el Q-learning necesita un guía?** El estado discreto no tiene posición
  ni memoria: el 32 % de las decisiones correctas eran indistinguibles con el
  estado v1. El guía aporta la planificación y la tabla aprende cuándo seguirlo
  ([auditoría](auditoria-qlearning-planetario.md#discretización)).
- **¿El guía hace trampa?** Conoce el mapa estático (rocas, grieta, cráteres), pero
  **nunca** la posición de las muestras: las descubre con la cámara. Un test lo
  verifica.
- **¿Por qué el PPO saca 100 % y aun así hay un límite?** Porque son 8 mapas fijos.
  En mapas procedurales cae a 50 % en L2 y 0 % en el resto: memoriza los mapas.
- **¿Cómo se recupera un modelo planetario?** El manifiesto da la URI dentro del
  commit exacto, su SHA-256, la normalización y el ambiente compatible.
  `python scripts/build_planetary_manifests.py --check` recalcula los hashes.
- **¿Qué pasa con lo aprendido en vivo?** Render gratuito no tiene disco persistente.
  La API arranca desde `qlearning-seed-v2`, y un checkpoint de una versión anterior
  se conserva renombrado, nunca se borra.

## Defensa de la entrega v1.0 (referencia)

## 1. Problema, misión, decisiones y límites del ambiente · Suleily

- **Misión:** B1 → B2 → base antes de agotar la batería. Hay que evitar la
  grieta, volcar y los choques.
- **Física:** MuJoCo en 3D (gravedad de Marte, dunas, grieta real).
  - Física a 50 Hz (dt 0.02 s); el agente decide a 2 Hz (0.5 s).
- **Contrato:**
  - observación de 20 floats (9 rayos, objetivo, velocidades, batería, etapa
    e inclinación) y `Discrete(5)`;
  - `terminated` para éxito, grieta, vuelco y batería;
  - `truncated` a los 400 pasos.
- **Niveles:** N1, N2 y N3, con la decisión nueva de cada uno
  ([ficha v1](ambiente-v1.md) y mapas).
- **Límites:** el mapa es fijo y los rayos no ven la grieta.

## 2. Baseline y evolución de la dificultad · Tomás

- **Referencias:** aleatorio 0 % en todos los niveles y regla 100 / 100 / 98 %.
- **Baseline aprendido:** Q-learning (exp-001), 97 ± 5 % en N2.
- Mostrar la tabla de [experimentos.md §3](experimentos.md#3-resultados) y la
  gráfica `evaluacion-por-nivel.png`.
- **Mensaje:** el retorno no es el éxito. Comparamos la tasa de éxito y los
  fallos por causa con la suite fija.

## 3. Hipótesis que mejoraron o empeoraron · Tomás y Nicol

- **exp-002, PPO frente a Q-learning:**
  - mismo éxito dentro de una desviación estándar;
  - PPO es mucho más eficiente (unos 85 pasos frente a 117–195) y estable;
  - test reservado: 100 % en las 3 semillas.
- **exp-003, currículo N1 → N2:** llega al 100 % con la mitad de los pasos
  en N2.
- **exp-004, más exploración:** no cambia nada, porque ya estaba en el techo.
  Hipótesis no confirmada.
- **exp-005, entrenar en N3:** 100 % en N3, pero 44 ± 48 % en N2. Rechazado
  para N2.
- **En la web:** *Comparar*, marcando `92cdc692`, `4d91d98f` y `3dd6e0f7`,
  con la suite N2 y luego la N3.

## 4. Demo desde Vercel: modelo histórico y recomendado · Natalia

1. *Catálogo*: filtrar por nivel. Abrir `4d91d98f` (Q-learning, primer run):
   - hipótesis, cambio, semilla, pasos, tiempo, commit (enlace a GitHub),
     MLflow, SHA-256 y decisiones;
   - la URL queda en `/run/4d91…`; recargar para mostrar que el enlace
     funciona.
2. *Evaluar en N2*: la identidad muestra el run, el ambiente, el nivel, dónde
   se entrenó y la configuración. Reproducir el episodio en 3D.
3. *Evaluar en N1* el mismo modelo: aparece el aviso "Evaluación en otro
   nivel" y se ve cómo cae en la grieta.
4. El recomendado de N2 (`92cdc692`, marcado con ★) en N2: 100 %.
5. *Control manual* con las flechas, a través de `/sim/step`.
6. Mostrar un error real: evaluar con un run inexistente o con la API caída.
   La web lo muestra como error; el replay se rotula `REPLAY`.

## 5. Incidente, recuperación y rollback · Nicol

- **Incidente:** [incidente.md](incidente.md).
  - Con las mismas semillas: 100 % → 0 %.
  - El bloqueo es la causa (el sensor solo, no): el rover gira frente a la
    roca hasta agotar la batería.
  - Ejemplo en la web: `92cdc692` en N3, semilla 300000.
- **Intervención:** exp-005, entrenado por el flow de Prefect (mostrar la
  evidencia JSON de las 8 etapas). Se acepta para N3 y se rechaza para N2,
  según `configs/selection.yaml`.
- **Rollback:**
  - mostrar `python pipelines/catalog.py history`: v3 *promote* (16 % en N2)
    → v4 *rollback* → `in_sync: true`;
  - en la web, *Monitoreo*: versión servida, éxito por escenario, causas de
    fallo y latencia y errores por ruta.
- **Si el docente lo pide en vivo:**
  `python pipelines/flow.py --rollback n2 --to <run anterior> --reason "..." --api-url <API>`.

## 6. Costo, limitaciones y siguiente hipótesis · Suleily

- **Costo:** 125 min de CPU en total; PPO unos 13 min por run; inferencia
  menor de 0.4 ms; servicios gratuitos.
- **Límites:** mapa fijo, techo de 100 % en N2, una sola semilla en
  exp-003 y exp-004, y la grieta invisible.
- **Siguiente hipótesis:** bloqueo en el 30 % de los episodios para tener un
  solo modelo para N2 y N3 ([informe §7](informe.md#7-siguiente-hipótesis-razonable)).

## Preguntas probables (todos deben poder responderlas)

### Contrato del ambiente

**¿Qué ve el agente? ¿Ve su posición?**
No ve su posición. Recibe 20 floats: 9 rayos de 0 a 5 m (de −90° a +90°), la
distancia y el seno y coseno del ángulo al objetivo, v y ω normalizadas, la
batería, la etapa en one-hot y el cabeceo y alabeo.

**¿Qué diferencia hay entre `terminated` y `truncated`?**
`terminated` es un fin propio de la misión: éxito, grieta, vuelco o batería.
`truncated` es el corte a los 400 pasos. Así el valor futuro se sigue
estimando cuando se acaba el tiempo.

**¿Por qué la etapa 3 es "volver a la base" también en N1?**
Para que la observación signifique lo mismo en los tres niveles y un modelo
se pueda evaluar en otro nivel sin adaptador.

**¿Se puede ganar puntos sin cumplir la misión?**
No, y hay pruebas para cada caso:
- los bonos se dan una vez y en orden;
- el progreso es una diferencia de distancias (ir y volver suma cero);
- no hay bono por sobrevivir;
- fallar cuesta −50.

**¿Cambia algo la velocidad de animación?**
No. La API calcula todos los cuadros con dt fijo; la web solo los dibuja más
rápido o más lento.

### Versionamiento y trazabilidad

**¿Cómo sé que el modelo que evalúo es el registrado?**
El manifiesto guarda el SHA-256 de los pesos. La API lo recalcula antes de
usarlo y responde 409 si no coincide (prueba
`test_tampered_model_is_rejected`).

**¿Qué identifica un run?**
- `run_id` de MLflow;
- commit;
- SHA-256 del config resuelto, del YAML fuente y del ambiente
  (`environment_hash`, código + nivel);
- suite con su SHA-256;
- semilla y presupuesto efectivo;
- `parent_run_id` si continúa de otro.

**¿Dónde están los pesos?**
- Punteros DVC en Git (`models/<run_id>.dvc`) y remoto en DagsHub.
- Copia pública en la release `models-v2` con `SHA256SUMS.txt`.
- La API desplegada los descarga de esa release y verifica el SHA-256.

**¿Qué pasa si se publica dos veces el catálogo?**
Con el mismo contenido, misma versión: `changed: false`. Con contenido
distinto, `catalog-vN+1` y una línea en `history.jsonl`. Nunca se sobrescribe
en silencio.

**¿Rollback es lo mismo que registrar un modelo nuevo?**
No. Rollback restaura un run que **ya fue** servido en ese nivel, y el código
rechaza cualquier otro. Promover uno nuevo es `promote`, que queda registrado
aparte.

**¿Por qué el recomendado de N2 es exp-004 y no exp-002?**
Por los criterios fijados antes: reemplazar al recomendado exige +5 puntos.
exp-004 se publicó primero y nadie lo supera (todos están en 100 %). Es una
limitación conocida: hay techo y falta una métrica secundaria.

### Desplegar y reproducir

**¿Cómo se despliega?**
- **API:** Render con `render.yaml` (plan gratuito, HTTPS). Sin torch: los PPO
  usan su adaptador numpy, validado con los mismos 150 casos por modelo.
- **Web:** Vercel (Root `web`, `VITE_API_BASE_URL` con la URL de Render).

**¿Cómo reproduzco una evaluación desde la release?**
1. Clonar el repositorio.
2. Bajar y extraer `robot-expedition-models-v2.zip`.
3. Correr `sha256sum -c SHA256SUMS.txt`.
4. Correr `python pipelines/evaluate.py --policy … --model models/<run_id>/… --level n2`.

Los 50 casos coinciden con `catalog/results/<run_id>/n2.json`
([prueba](evidencia/rollback.md#3-reconstrucción-desde-un-clon-limpio)).

**¿Cómo se corre el pipeline?**
`python pipelines/flow.py --config configs/experiments/exp-00X.yaml` (Prefect
3.4.20). Las 8 etapas dejan evidencia en `docs/evidencia/pipeline/`.

**¿El adaptador numpy no cambia el modelo?**
No. Usa los mismos pesos y la misma normalización, y reproduce los 150 casos de
cada PPO en las tres suites ([evidencia](evidencia/adaptador-numpy.json)). Además
guarda el SHA-256 del modelo original y la API lo compara con el manifiesto.

**¿Por qué no se ejecuta la red en el navegador?**
Exportar la red no traslada el ambiente. La API usa el mismo MuJoCo y las
mismas clases de política del entrenamiento.

**¿Dónde está el test reservado?**
Sus semillas no están en Git (`EXPEDITION_TEST_SEEDS` o
`test-final.local.yaml`, ignorado). Se usó una vez por finalista y los
resultados no publican las semillas.
