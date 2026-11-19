import os

# Las pruebas no deben descargar el checkpoint Q-learning base desde GitHub al importar api.main.
os.environ["PLANETARY_Q_SEED_URL"] = ""
