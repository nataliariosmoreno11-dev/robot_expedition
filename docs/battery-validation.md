# Autonomía de la campaña

La validación física de los diez escenarios fijos completó todas las misiones,
incluyendo recolección y regreso, con el presupuesto original y cero colisiones.
Los resultados, fecha y hash del catálogo están en
`web/public/demo/battery-audit.json` y se muestran en la página.

Se utilizó el seguidor de rutas de referencia de `scripts/campaign.py`, separado
del PPO. La prueba no modifica pesos ni el currículo y no demuestra aprendizaje.
Se ejecutó un intento por escenario con semilla 700001; demuestra factibilidad
física en esos intentos, no una tasa de éxito garantizada con cualquier semilla.

| Escenario | Batería restante |
|---|---:|
| 1 | 92,7 % |
| 2 | 53,8 % |
| 3 | 45,0 % |
| 4 | 72,0 % |
| 5 | 61,0 % |
| 6 | 59,4 % |
| 7 | 58,4 % |
| 8 | 28,6 % |
| 9 | 27,9 % |
| 10 | 36,3 % |

El PPO aún no supera el escenario 6. El margen físico observado indica que
no es necesario aumentar la capacidad para hacer posible esa misión.
Los fallos por batería deben analizarse junto con los choques y la longitud del
recorrido. Aumentar la batería cambiaría la tarea y haría menos comparables las
evaluaciones históricas, por lo que esta revisión conserva la configuración.

Reproducir la validación desde la raíz:

```sh
.venv/bin/python scripts/campaign.py validate --physical --output logs/battery-validation.json
```

La visualización permite abrir «Comparar versiones del modelo», escoger una
versión y escenario, buscar intentos y reproducir uno. Los intentos conservan
éxitos y fallos: son replays históricos, no inferencias nuevas con los pesos
finales de la versión. El botón de pantalla completa conserva estos controles y
los de reproducción; Escape permite salir.
