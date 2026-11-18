# Agente automático de la campaña

La web local permite entrenar PPO, detenerlo conservando el checkpoint,
continuar desde una versión y ejecutar su política congelada en cualquiera de
las diez misiones. Arranque:

```bash
ENABLE_LOCAL_TRAINING=1 POLICY_BACKEND=numpy .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
cd web
VITE_API_BASE_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0
```

`POLICY_BACKEND=numpy` corresponde a los modelos históricos de balizas; el
entrenador planetario usa Stable-Baselines3 y PyTorch CPU. Habilitar entrenamiento
es una opción del servidor local, no del despliegue público.

## Qué aprende

PPO (`MlpPolicy`) aprende una distribución sobre siete acciones: avanzar,
avanzar girando a izquierda/derecha, girar a izquierda/derecha, recolectar y
retroceder. La entrada son cuatro observaciones consecutivas (132 valores),
normalizadas con VecNormalize. Las observaciones incluyen sensores y recursos;
la política no recibe rutas de referencia ni posiciones de muestras ocultas.
La recompensa proviene del ambiente: recolección correcta, exploración y regreso;
colisiones, energía y recolección incorrecta penalizan. Una recompensa creciente
por sí sola no demuestra que resolvió las misiones.

La configuración original es `configs/experiments/exp-007-planetary.yaml`.
PPO optimiza una mejora limitada de la política (`clip_range=0.2` por defecto)
con tasa de aprendizaje 0.0003, descuento 0.99, GAE 0.95 y entropía 0.015.
La interfaz usa `exp-009-retention.yaml`: tasa 0.00005, clip PPO 0.1,
`target_kl=0.015`, entropía 0.02 y 50% de repaso de misiones previas.
Guarda checkpoints cada 4.096 pasos y evalúa dominio cada 16.384.
El ensayo `exp-008` aumentó la exploración pero perdió la misión 1 en la
comparación de políticas congeladas; se conserva como evidencia, no recomendado.
Reinicia solo los intentos de entrenamiento con 80 pasos sin explorar una celda
nueva ni recolectar (incluye recolectar inmóvil y ciclos de giros). Se registra
`sin_progreso` como truncamiento, nunca como éxito; las evaluaciones y el control
manual conservan la duración original. No se añaden rutas ni objetivos ocultos.
Las métricas `training/explored_cells` y `training/no_progress_steps` permiten
comprobar si el agente explora. Al reanudar se aplican los nuevos hiperparámetros
al modelo cargado, conservando pesos, optimizador y normalización.
Un control de retención cada 4.096 pasos evalúa las misiones dominadas al inicio
del run con ocho semillas fijas independientes. Exige al menos 7/8 éxitos y
no más de dos colisiones medias. Si falla, guarda el candidato rechazado y
restaura pesos, optimizador y normalización del último checkpoint aceptado.
Este control previene regresiones; no sustituye las 50 pruebas para certificar
dominio. Al detener se conservan los últimos pesos aceptados. El contador
incluye también los pasos de actualizaciones rechazadas.
La interfaz usa rollouts de 256 pasos por ambiente (4 ambientes, 1.024 pasos por
actualización) para poder detener y guardar con frecuencia. Por ello PPO puede
redondear un presupuesto al tamaño del rollout. Se conservan los hiperparámetros
efectivos en cada run. La promoción exige las evaluaciones independientes del
currículo existente (al menos 50 episodios y 85% de éxito, retención incluida).

## Uso

La configuración activa es ahora `exp-010-parallel.yaml`: conserva los parámetros
de aprendizaje y retención de exp-009, pero ejecuta los cuatro entornos en procesos
separados. `scripts/benchmark_vector_env.py` compara semillas, acciones, transiciones
y registro completo de pasos. En esta máquina midió 171 frente a 190 pasos/s,
con transiciones idénticas en precisión float32; el arranque de los procesos cuesta
unos 8,6 segundos. El informe está en `docs/evidencia/vector-throughput.json`.

Las evaluaciones curriculares rechazan anticipadamente un candidato cuando ni
acertando todos los casos restantes podría alcanzar el umbral. Por ejemplo, ocho
fallos hacen imposible superar 85% en 50 episodios. El informe queda marcado parcial
y no permite promoción. Los candidatos aprobados siguen necesitando todos los casos;
la evaluación final de publicación conserva sus 500 episodios.

La protección de retención incorpora inmediatamente cada escenario recién aprobado
y recupera juntos los pesos, la normalización y el estado curricular. Al reanudar,
exp-010 comprueba las habilidades declaradas: si el checkpoint final las perdió,
busca el último checkpoint de dominio que las conserve. Registra esa elección y
sus evaluaciones en `resume-selection.json`. El contador puede retroceder al paso
del checkpoint recuperado; los intentos posteriores permanecen en el run anterior.

Para reanudar una campaña detenida, `python -m api.campaign_worker --parent-job <id>
--resume` continúa desde el checkpoint interrumpido registrado por el supervisor,
conserva su historial y crea una nueva versión.

### Campaña automática en segundo plano

`python -m api.campaign_worker --parent-job <job_id>` encadena runs de 262.144
pasos adicionales desde los pesos guardados. No necesita navegador ni renderizado.
El proceso debe lanzarse desacoplado de la terminal; el estado queda en
`logs/autonomous_campaign/state.json`. Al relanzarlo recupera el run activo o el
último padre, sin crear un segundo supervisor. Sigue mientras el equipo esté
encendido; no instala un servicio de arranque ni garantiza convergencia.

Cada run conserva checkpoints, episodios, pasos, configuración, código, métricas
y versiones DVC/MLflow. Cuando el currículo ha superado los diez escenarios, una
misma política congelada realiza otras 500 evaluaciones (50 por escenario). Sólo
se publica si **cada** escenario alcanza al menos 85% de éxito y como máximo dos
colisiones medias. Esta prueba usa semillas distintas de las evaluaciones
curriculares y conserva las reglas originales; certifica estos escenarios fijos,
no mapas arbitrarios.

La publicación es atómica en `models/final-release.json`, con hashes y referencia
al informe `models/<run>/public-evaluation.json`. Los pasos de la evaluación final
se incluyen en la instantánea SQLite y DVC. Todos los visitantes de esta instancia
ven el mismo botón **Ver modelo final validado**; antes de aprobar el control,
la interfaz dice explícitamente que todavía no hay modelo final. Los replays de
entrenamiento están desactivados inicialmente. El acceso por Internet requiere
desplegar esta aplicación y sus artefactos; ejecutar localmente no los publica
en un servidor remoto.

`GET /training/campaign` muestra avance y publicación. **Detener campaña automática
y guardar** (`POST /training/campaign/stop`) detiene el supervisor y solicita guardar
el worker. También se puede crear `logs/autonomous_campaign/stop`. Para reanudar
tras una detención explícita hay que retirar ese archivo. Si falla un run o quedan
menos de 5 GiB libres, conserva los artefactos y registra el error para intervención;
no lo presenta como un modelo terminado.

1. **Entrenar desde cero** crea una política nueva y un run MLflow.
2. La web reproduce automáticamente intentos guardados del agente, rotulados
   REPLAY. El entrenamiento sucede en Python aunque cierres la pestaña.
3. **Detener y guardar** espera a terminar la actualización PPO actual. No es
   una pausa instantánea. Durante la evaluación final se interrumpe la evaluación
   y se conservan los pesos ya entrenados.
4. **Continuar pesos** crea otro run, con referencia al padre, y carga pesos,
   normalización, optimizador, contador y estado curricular. No sobrescribe el
   run anterior.
5. **Ver política guardada en este escenario** ejecuta inferencia determinista
   en el simulador, sin aprendizaje ni modificación de los pesos.

## Evidencia y versionado

Los checkpoints de 396.288 y 461.824 pasos están disponibles en la
[release preliminar planetary-checkpoints-20261114](https://github.com/nataliariosmoreno11-dev/robot_expedition/releases/tag/planetary-checkpoints-20261114).
El ZIP contiene pesos, normalización, optimizador, configuración y evaluaciones;
la descarga pública se verificó contra su SHA-256. No es el modelo final ni incluye
las bases SQLite completas. El entrenamiento posterior sigue siendo local y no se
sube automáticamente. `scripts/package_training_weights.py` reproduce el paquete.
Estos modelos usan el contrato planetario de 132 observaciones y 7 acciones;
no son intercambiables con el endpoint histórico de balizas `/evaluate`.

- `logs/training_jobs/<job>/`: configuración solicitada, estado y registro del proceso.
- `models/<run>/`: pesos PPO, normalización, optimizador, contador, currículo,
  configuración efectiva, hashes SHA-256 y snapshot del código fuente usado.
- `models/<run>/checkpoint-<pasos>/`: checkpoints periódicos inmutables tras
  actualizaciones PPO (cada 4.096 pasos en la configuración de la web y al detener).
  `retention-safe/` contiene la política aceptada y `rejected-<pasos>/` conserva
  los candidatos que olvidaron misiones, junto a sus evaluaciones.
- `models/<run>.dvc`: puntero DVC de los artefactos, también para runs interrumpidos.
- `manifests/<run>.json`: estado, hashes, commit, padre, presupuesto y evaluación.
- `mlflow.db` y `mlruns/`: parámetros, métricas, artefactos y versiones del modelo.
- `logs/planetary_episodes.sqlite`: todos los intentos y pasos, incluidos fallos,
  sin poda de la campaña. Los replays no se regeneran con una política posterior.
  Al cerrar un run se copia una instantánea consistente de esta base a
  `models/<run>/episodes.sqlite`, incluida en su versión DVC.

Los checkpoints se verifican por SHA-256 antes de continuar o servirlos. El
versionado DVC es local: publicar la copia remota requiere `dvc push` y las
credenciales del equipo; la interfaz no sube artefactos ni hace commits.

Un entrenamiento corto verifica el flujo técnico. Para afirmar dominio se deben
revisar las evaluaciones por escenario, éxito, colisiones y energía. No se promete
que un presupuesto concreto produzca dominio de las diez misiones.

La web sigue el entrenamiento activo y, al terminar, selecciona la mejor versión
con evaluación final disponible. No asume que más pasos significan mejor política.
Se puede seleccionar una versión histórica explícitamente para comparar.

## Resultado del entrenamiento con retención

El run `e24bfc5140734088b9b1d76fc14cca15` terminó con 101.376 pasos
acumulados (65.536 adicionales al padre `cfc7a1fb63ef4c91a1afb1e0e6828e3a`).
El guardián rechazó la actualización del paso 48.128 y restauró el último
checkpoint aceptado; las comprobaciones posteriores conservaron el nivel 1.

La comparación de políticas congeladas usa las mismas tres semillas
120001–120003 y las reglas originales de evaluación, sin el corte de entrenamiento
por falta de progreso. Los resultados completos, con hashes de los pesos, están en
[`training-retention-review.json`](evidencia/training-retention-review.json).

| Métrica | Padre | Nueva versión |
| --- | ---: | ---: |
| Nivel 1: misiones completas | 3/3 | 3/3 |
| Nivel 1: pasos medios | 46 | 11 |
| Nivel 2: muestras medias recogidas | 0/2 | 1/2 |
| Nivel 2: misiones completas | 0/3 | 0/3 |

Es una mejora de eficiencia en el primer escenario y de recolección en el segundo,
pero todavía no resuelve el nivel 2: termina por batería. La evaluación final de
24 episodios repartidos entre los diez niveles obtuvo 12,5% de éxito global y sólo
éxitos en el nivel 1. Tres semillas en escenarios fijos no demuestran generalización
a mapas nuevos ni sustituyen el criterio curricular de dominio.
