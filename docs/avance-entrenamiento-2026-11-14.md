# Avance guardado de entrenamiento

El checkpoint `12e968e7e89048b3b5609aac9301d855` contiene 2.558.976 pasos
acumulados, escenarios 1–5 aprobados por el currículo y el escenario 6 pendiente.
No es el modelo final certificado de los diez escenarios.

Su evaluación general de 24 episodios obtuvo 58,33 % de éxito: éxitos en los
niveles 1–5 y ninguno en 6–10. Registró 180,54 colisiones medias por episodio;
por tanto, los resultados todavía requieren mejoras y validación más amplia.
El historial contiene regresiones; más pasos no garantizan conservar dominio.

Los manifiestos y punteros DVC se guardan en esta rama. Los pesos, normalización,
optimizador, configuración y evaluaciones de cuatro versiones recientes están
en la [release de avances](https://github.com/nataliariosmoreno11-dev/robot_expedition/releases/tag/training-progress-20261114).
El archivo incorpora sumas SHA-256 y se extrae en la raíz del repositorio.

```sh
.venv/bin/python scripts/package_training_weights.py --runs \
  5ae5623bf2954084b8ba2c9833abbf89 \
  6a7190d0286842e9bea203e07d31a5ce \
  a25e4a0a6f164b4c88aafc976be827f6 \
  12e968e7e89048b3b5609aac9301d855
```

Las bases de episodios, registros completos, MLflow y checkpoints intermedios
permanecen locales. Los punteros DVC no prueban que esos archivos estén disponibles
en el remoto DVC; la descarga publicada incluye únicamente los archivos del ZIP.
Las credenciales SMTP y direcciones destinatarias permanecen en configuración local.

Estos pesos usan 132 observaciones y siete acciones: requieren el entorno
planetario y su normalización, no el endpoint histórico de balizas `/evaluate`.
Publicarlos no actualiza automáticamente el servidor desplegado.
