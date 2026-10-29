"""Carga y validación del dataset de ventas de vehículos.

Si el CSV no existe (por ejemplo, en una clonación nueva), se regenera con
generate_data.py usando la semilla de configs/config.yaml.
"""
import hashlib
from pathlib import Path

import pandas as pd

from src.config import ruta_absoluta
from src.generate_data import generate_car_sales_data


def huella_archivo(ruta: str | Path) -> str:
    """MD5 del archivo: permite auditar con qué versión exacta de los datos se entrenó."""
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def validar_columnas(df: pd.DataFrame, columnas_esperadas: list[str]) -> None:
    """Lanza un error si faltan columnas esperadas."""
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")


def cargar_datos(config_datos: dict) -> pd.DataFrame:
    """Lee el CSV indicado en la configuración, lo genera si no existe y valida sus columnas."""
    ruta = ruta_absoluta(config_datos["ruta"])
    if not ruta.exists():
        generate_car_sales_data(
            num_records=config_datos["num_registros"],
            random_state=config_datos["semilla"],
            output_path=ruta,
        )
    df = pd.read_csv(ruta, parse_dates=["fecha_venta"])
    validar_columnas(df, config_datos["columnas_esperadas"])
    if df.empty:
        raise ValueError("El dataset está vacío")
    return df
