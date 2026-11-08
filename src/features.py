"""Selección de las variables del experimento según configs/features.yaml."""
from dataclasses import dataclass

SELECCIONES = ("ALL", "NUMERIC", "CUSTOM")


@dataclass(frozen=True)
class Variables:
    alias: str
    categoricas: list[str]
    numericas: list[str]

    @property
    def todas(self) -> list[str]:
        return self.categoricas + self.numericas

    @property
    def n(self) -> int:
        return len(self.todas)


def resolver_variables(config_features: dict, columnas_dataset: list[str] | None = None) -> Variables:
    """Devuelve las variables a usar según el alias de `seleccion` (ALL, NUMERIC o CUSTOM).

    Si se pasan las columnas del dataset, comprueba que todas las variables existan.
    """
    alias = str(config_features["seleccion"]).upper()
    numericas = list(config_features["numericas"])
    categoricas = list(config_features["categoricas"])

    if alias == "ALL":
        elegidas = Variables(alias, categoricas, numericas)
    elif alias == "NUMERIC":
        elegidas = Variables(alias, [], numericas)
    elif alias == "CUSTOM":
        custom = list(config_features.get("custom") or [])
        if not custom:
            raise ValueError("La selección CUSTOM necesita al menos una variable en `custom`")
        desconocidas = [c for c in custom if c not in numericas + categoricas]
        if desconocidas:
            raise ValueError(f"Variables de `custom` que no están en numericas ni categoricas: {desconocidas}")
        # Conserva el orden de features.yaml y separa cada variable según su tipo.
        elegidas = Variables(alias, [c for c in custom if c in categoricas], [c for c in custom if c in numericas])
    else:
        raise ValueError(f"Selección no válida: {alias}. Opciones: {SELECCIONES}")

    if columnas_dataset is not None:
        faltantes = [c for c in elegidas.todas if c not in columnas_dataset]
        if faltantes:
            raise ValueError(f"Variables que no están en el dataset: {faltantes}")
    return elegidas
