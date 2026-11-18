# Auditoría del aprendizaje planetario — 18 de noviembre de 2026

## Alcance y conservación

La auditoría no modifica ni elimina checkpoints de Tomás. La evidencia aprobada
de la campaña fija y sus ocho modelos permanece intacta. Los cambios nuevos
usan un experimento y rutas de salida diferentes.

## Hallazgos reproducibles

1. El entorno planetario expone 33 variables por instante y los PPO usan cuatro
   instantes (132 entradas), siete acciones discretas, normalización `VecNormalize`
   y una MLP 132→64→64 con ramas de política y valor.
2. La política denominada Q-learning interactivo no es autónoma: el checkpoint
   actual registra 4.424 decisiones guiadas, 82 aleatorias y 1.812 greedy. La
   mayoría del comportamiento observado procede del controlador reactivo.
3. Q-learning tabular reduce el estado a once categorías. Estados diferentes de
   mapas, posiciones y misiones pueden colisionar en la misma clave, lo que
   explica acciones contradictorias y mala transferencia.
4. El corte de episodio por recompensa +20 no representa éxito y podía finalizar
   antes de completar recolección y retorno. Se retiró de la sesión interactiva.
5. Render guardaba la memoria en `/tmp/robot-expedition`; ese almacenamiento es
   efímero. El blueprint queda preparado para un disco montado en `/var/data`.
6. Los PPO aprobados tienen 100 % en los ocho escenarios fijos, pero esa cifra no
   prueba generalización procedural.

## Diagnóstico procedural inicial

Comando reproducible:

```bash
python scripts/diagnose_planetary_learning.py --episodes 2 \
  --seed-start 970000 --q-checkpoint <checkpoint-activo>
```

La evidencia completa está en
`docs/evidencia/diagnostico-aprendizaje-procedural.json`.

- Política aleatoria: 0 % de éxito en L1–L8.
- Q-learning greedy sin guía: 0 % en L1–L8, tres celdas exploradas de media y
  estancamiento en todos los episodios.
- PPO de campaña sobre mapas procedurales: 50 % en L2 (una de dos semillas) y
  0 % en los demás niveles de esta prueba pequeña. No es una estimación final,
  pero rechaza la afirmación de generalización completa.
- Los bloqueos dominantes son choques repetidos, giro sin desplazamiento,
  oscilación y falta de exploración. Ahora se registran por separado.

## Causa principal

El Q tabular pierde información espacial y temporal, mientras el PPO anterior
fue optimizado y seleccionado sobre escenarios fijos. El problema combina
representación insuficiente y cambio de distribución; no se resuelve aumentando
recompensas o reiniciando antes.

Además, el área activa procedural era menor que el terreno físico y su frontera
no aparecía en el LIDAR. Para la política, continuar recto parecía válido, pero
el entorno terminaba el episodio por `fuera_area`. Esto violaba el supuesto de
observación Markoviana. La frontera ahora limita los rayos procedurales; la
campaña fija y las observaciones de los checkpoints de Tomás no cambiaron.

## Experimento prioritario

`exp-016-procedural-generalization` entrena PPO desde cero solo en L1 procedural,
sin reglas, con cuatro observaciones temporales y potencial basado exclusivamente
en objetivos visibles o retorno conocido. Usa semillas distintas para entrenar y
validar, evalúa cada 4.096 pasos y conserva candidatos separados.

No se avanzará a L2 hasta demostrar mejora en misión completa, muestras,
retorno, colisiones, cobertura y estancamiento sobre semillas reservadas.

## Resultado de la primera etapa L1

El primer intento de 4.096 pasos produjo un checkpoint, pero el pipeline marcó
el run como fallido porque DVC no estaba instalado. El registro ahora degrada a
una URI local después de guardar checkpoint y artefactos, en lugar de invalidar
un entrenamiento terminado.

Después de corregir el LIDAR, `exp-016` llegó a 35 % de éxito en 20 semillas
reservadas a los 8.192 pasos. `exp-017` reanudó el mismo checkpoint, incluyendo
optimizador y `VecNormalize`, y evolucionó así:

| Pasos acumulados | Éxito | Estancamiento medio | Choques/episodio |
| ---: | ---: | ---: | ---: |
| 12.288 | 65 % | 50,00 | 0,00 |
| 16.384 | 65 % | 60,45 | 9,10 |
| 20.480 | 75 % | 29,15 | 0,00 |
| 24.576 | 80 % | 16,35 | 0,85 |

La evaluación final separada obtuvo 91,67 % en 12 episodios. Para evitar aceptar
el modelo por una muestra pequeña, se ejecutó otra evaluación de 50 episodios
con las mismas semillas para todas las políticas. Evidencia:
`docs/evidencia/comparacion-independiente-l1.json`.

| Política | Éxito | Retorno medio | Choques/episodio | Pasos estancado |
| --- | ---: | ---: | ---: | ---: |
| Aleatoria | 4 % | -35,21 | 21,00 | 31,42 |
| Q-learning anterior sin guía | 0 % | -118,61 | 193,00 | 184,00 |
| PPO de Tomás fuera de campaña | 0 % | -37,08 | 15,18 | 101,36 |
| PPO procedural mejorado | **82 %** | **212,35** | **0,32** | **25,14** |

El hito L1 queda aceptado por evidencia independiente. Esto no demuestra dominio
de L2–L8; esos niveles deben incorporarse mediante currículo y conservar L1 con
una prueba de retención antes de promover cada checkpoint.
