"""Lectura de la configuración del proyecto (configs/config.yaml)."""
from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ_PROYECTO / "configs" / "config.yaml"


def cargar_config(ruta: str | Path = RUTA_CONFIG) -> dict:
    """Devuelve el contenido de config.yaml como diccionario."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta_absoluta(ruta_relativa: str | Path) -> Path:
    """Convierte una ruta relativa a la raíz del repositorio en absoluta."""
    ruta = Path(ruta_relativa)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta
