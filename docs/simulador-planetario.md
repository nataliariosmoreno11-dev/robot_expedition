# Simulador planetario

La campaña actual reemplaza los mapas procedurales de entrenamiento por diez
misiones fijas. Su guía completa está en [Campaña de diez escenarios](campana-diez-escenarios.md).

- [Contrato de observación, acciones, almacenamiento y API](contratos-planetario.md).
- [Configuración PPO](../configs/experiments/exp-007-planetary.yaml).
- [Definiciones independientes](../configs/campaign/).
- [Evidencia de validación física](evidencia/campaign-validation.json).

El generador procedural anterior se conserva como herramienta experimental,
pero no controla los mapas de la campaña. La antigua retención de diez replays
se sustituye por registro completo de todos los intentos de campaña.
