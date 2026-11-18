# Experimento 011: transferencia estable al escenario 6

La versión `da892feee1e94ecca1019c833754d1bb` registró 51 retrocesos en 64
controles de retención. El escenario 6 seguía con cero éxitos. El contador total
incluye esos pasos descartados: no mide cuánto aprendizaje se conserva.

La nueva configuración es `configs/experiments/exp-011-stable-transfer.yaml`.
Se prueba desde los pesos existentes, sin modificar mapas, batería ni los
requisitos de aprobación:

- Congelar las estadísticas de normalización heredadas evita que cambie la
  escala de entrada mientras se adapta la red. Es una hipótesis para reducir
  olvido, no una mejora de rendimiento demostrada por sí sola.
- Tasa de aprendizaje de 0,00002, límite KL de 0,01, rollouts de 512 pasos por
  entorno y lotes de 128 para actualizaciones más conservadoras.
- Gamma de 0,995 para dar más peso a resultados lejanos en misiones de varias
  muestras. Entropía de 0,01; se mantiene exploración estocástica PPO.
- Tras la prueba inicial, el 60 % de los intentos repasa escenarios anteriores.
  El escenario 3 tiene peso 4 y los otros peso 1: recibe aproximadamente el 30 %
  de todos los intentos mientras se aprende el 6; el 6 recibe el 40 %.
- Control de retención cada 16.384 pasos y al terminar. Los pesos se restauran
  si se olvidan escenarios; los criterios de retención no se relajan.
- El reinicio exclusivo de entrenamiento pasa de 80 a 120 pasos sin progreso
  para permitir rodeos más largos. La evaluación no aplica ese reinicio.

La prueba inicial tiene 16.384 pasos adicionales. Sus resultados deben
compararse con las evaluaciones del padre: éxitos, muestras, choques y causas
de fallo, además de retrocesos. No se afirma aprobación hasta pasar la suite
independiente de al menos 50 episodios y los controles de retención.

Resultado de la prueba inicial `63a5e57154cc48329dd5c62afc43e526`:
0/8 éxitos en el escenario 6, 2,125 muestras de media y máximo 3/4, con 92,125
colisiones de media. Retuvo los niveles 1, 2, 4 y 5 (8/8, sin choques), pero
falló en el 3 (5/8 y 330,125 colisiones de media). Se restauraron los pesos
seguros. La política candidata queda en `rejected-3378176`; no se adopta como
modelo aprobado. El repaso ponderado es el siguiente ajuste experimental,
pendiente de evaluación, no una mejora ya demostrada.

## Almacenamiento

Se conservan los dos checkpoints intermedios más recientes, un checkpoint
rechazado, los pesos raíz y todos los hitos de dominio/retención. Los resúmenes
de evaluaciones siguen guardados. Los replays mantienen la política compacta y
no se vuelve a copiar toda la base SQLite por versión.

La campaña solicita detener y guardar si Linux o Windows C: tienen menos de
10 GiB libres. El espacio virtual de WSL no sustituye al espacio real de C:.
