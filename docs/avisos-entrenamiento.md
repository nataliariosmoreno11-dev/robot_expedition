Los avisos por SMTP se generan solo ante la primera aprobación guardada de cada
escenario. Al iniciar el monitor, las aprobaciones existentes se consideran la
base y no se envían retrospectivamente. Las regresiones y nuevas aprobaciones
del mismo escenario no generan otro aviso. Cada destinatario se entrega por
separado y los envíos fallidos quedan pendientes para reintentar.

La configuración privada está en `logs/training_notifications/config.json`
(ignorada por Git). Completar localmente `sender`, `username` y `password` con
una cuenta SMTP autorizada; `host`, `port` y `tls` deben corresponder al proveedor.
No introducir secretos en el chat ni subir este archivo al repositorio.
El monitor vuelve a leer la configuración cada 30 segundos.

Ejecutar desde WSL con el Python del proyecto:

```sh
.venv/bin/python scripts/training_notifications.py
```

Para operación independiente de la terminal debe iniciarse con una sesión
separada y entrada/salida redirigidas. Una exclusión de archivo impide dos
monitores simultáneos. El estado está en
`logs/training_notifications/state.json`: `waiting_for_smtp` indica que falta
configurar el envío; `monitoring` indica configuración presente, y
`delivery_error` indica un intento fallido. `sent` registra aceptación por SMTP,
no entrega confirmada a la bandeja del destinatario.

Los eventos incluyen escenario, evaluación y ruta al checkpoint. No se envían
pesos ni avisos al terminar rondas. La aprobación curricular no equivale a la
certificación final de los diez escenarios. Los avisos requieren que el equipo
y WSL permanezcan funcionando. Una caída justo después del envío pero antes de
guardar su confirmación puede producir un reintento duplicado.
