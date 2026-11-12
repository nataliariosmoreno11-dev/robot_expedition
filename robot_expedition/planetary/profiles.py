"""Perfiles físicos usados por la simulación.

Los valores astronómicos (gravedad y densidad atmosférica aproximada) están
separados de los coeficientes de ingeniería del rover. Estos últimos son
simplificaciones explícitas y se usan para producir efectos observables en
MuJoCo; no pretenden describir todo el regolito marciano.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlanetProfile:
    planet_id: str
    display_name: str
    gravity_m_s2: float
    atmosphere_density_kg_m3: float
    reference_temperature_c: float
    terrain_friction: float
    rolling_resistance: float
    wind_drag_coefficient: float = 1.2
    rover_frontal_area_m2: float = 0.12

    def to_dict(self) -> dict:
        return asdict(self)


MARS = PlanetProfile(
    planet_id="mars",
    display_name="Marte",
    gravity_m_s2=3.71,
    atmosphere_density_kg_m3=0.020,
    reference_temperature_c=-55.0,
    terrain_friction=0.82,
    rolling_resistance=0.16,
)


_PLANETS = {MARS.planet_id: MARS}


def get_planet(planet_id: str = "mars") -> PlanetProfile:
    """Devuelve un perfil registrado; agregar otro planeta no cambia el ambiente."""
    try:
        return _PLANETS[planet_id]
    except KeyError as exc:
        raise ValueError(f"planeta no soportado: {planet_id!r}") from exc
