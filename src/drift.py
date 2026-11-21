"""Lógica de drift: emulación, validación de lotes y detección.

No usa Prefect ni MLflow, así que se prueba con pytest sin servicios levantados.

Emulación del drift (actividad 3)
---------------------------------
El nivel (0.0 a 1.0) decide cuántas de las 5 variables de entrada cambian:
`int(nivel × 5)`. Las variables se afectan siempre en este orden, cada una con un cambio
que imita algo que podría pasar en el mercado de vehículos usados:

    1. kilometraje   × 1,6          llegan vehículos más usados
    2. año_modelo    − 4 años       llegan modelos más antiguos
    3. tipo_vehiculo 50 % → Pick-up cambia la mezcla de carrocerías
    4. marca         50 % → Ford    cambia la mezcla de marcas
    5. precio_lista  × 1,3          los precios publicados suben

Así, 0.0 no cambia nada, 0.3 cambia 1 variable (20 %), 0.6 cambia 3 (60 %) y 1.0 las 5.
Además, el precio de venta (la variable objetivo) se multiplica por (1 + nivel × 0,25):
+7,5 % con 0.3 y +15 % con 0.6, como una inflación de los precios de cierre.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

N_VARIABLES = 5
SUBIDA_MAXIMA_OBJETIVO = 0.25  # con nivel 1.0, el precio de venta sube un 25 %


def _mas_kilometraje(df, rng):
    df["kilometraje"] = (df["kilometraje"] * 1.6).round().astype(df["kilometraje"].dtype)


def _modelos_mas_antiguos(df, rng):
    df["año_modelo"] = df["año_modelo"] - 4


def _mas_pickups(df, rng):
    filas = rng.random(len(df)) < 0.5
    df.loc[filas, "tipo_vehiculo"] = "Pick-up"


def _mas_ford(df, rng):
    filas = rng.random(len(df)) < 0.5
    df.loc[filas, "marca"] = "Ford"


def _precios_lista_mas_altos(df, rng):
    df["precio_lista"] = (df["precio_lista"] * 1.3).round(2)


# Orden fijo en que el drift va afectando las variables.
EFECTOS = [
    ("kilometraje", _mas_kilometraje),
    ("año_modelo", _modelos_mas_antiguos),
    ("tipo_vehiculo", _mas_pickups),
    ("marca", _mas_ford),
    ("precio_lista", _precios_lista_mas_altos),
]


def variables_afectadas(nivel: float) -> list[str]:
    """Variables que cambia un nivel de drift: las primeras int(nivel × 5) de EFECTOS."""
    validar_nivel(nivel)
    n = math.floor(nivel * N_VARIABLES + 1e-9)
    return [nombre for nombre, _ in EFECTOS[:n]]


def validar_nivel(nivel: float) -> None:
    if not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe estar entre 0 y 1, no {nivel}")


def aplicar_drift(df: pd.DataFrame, nivel: float, objetivo: str = "precio_venta", semilla: int = 42) -> pd.DataFrame:
    """Devuelve una copia del lote con el drift emulado. Con nivel 0.0 el lote no cambia."""
    validar_nivel(nivel)
    lote = df.copy()
    rng = np.random.default_rng(semilla)
    afectadas = set(variables_afectadas(nivel))
    for nombre, efecto in EFECTOS:
        if nombre in afectadas:
            efecto(lote, rng)
    if nivel > 0:
        lote[objetivo] = (lote[objetivo] * (1 + nivel * SUBIDA_MAXIMA_OBJETIVO)).round(2)
    return lote


def dividir_entrenamiento_prueba(df: pd.DataFrame, test_size: float, semilla: int):
    """Parte los datos igual que el entrenamiento (mismas filas). Devuelve (entrenamiento, prueba)."""
    return train_test_split(df, test_size=test_size, random_state=semilla)


def tomar_lote(prueba: pd.DataFrame, tamano: int, semilla: int) -> pd.DataFrame:
    """Muestra de la parte de prueba: vehículos que el modelo no vio al entrenar."""
    if tamano > len(prueba):
        raise ValueError(f"El lote ({tamano}) no puede ser mayor que la parte de prueba ({len(prueba)})")
    return prueba.sample(n=tamano, random_state=semilla).reset_index(drop=True)


def validar_lote(lote: pd.DataFrame, objetivo: str = "precio_venta") -> None:
    """Falla con un mensaje claro si el lote no trae las etiquetas reales."""
    if objetivo not in lote.columns:
        raise ValueError(f"El lote no tiene la columna '{objetivo}': sin etiquetas reales no se puede "
                         "calcular el drift del target. En producción llegan con retraso; espera a tenerlas.")
    if lote[objetivo].isna().any():
        faltan = int(lote[objetivo].isna().sum())
        raise ValueError(f"Al lote le faltan {faltan} valores de '{objetivo}': no se puede calcular el drift del target.")


def drift_de_target(referencia: pd.Series, lote: pd.Series, alfa: float = 0.05) -> dict:
    """Compara la distribución del precio de venta del lote con la del entrenamiento.

    Usa la prueba de Kolmogorov-Smirnov de dos muestras, porque el objetivo es continuo
    (en clasificación se compararía una tasa con chi-cuadrado).
    """
    estadistico, p_valor = ks_2samp(referencia, lote)
    media_ref, media_lote = float(referencia.mean()), float(lote.mean())
    return {
        "media_referencia": media_ref,
        "media_lote": media_lote,
        "cambio_media": (media_lote - media_ref) / media_ref,
        "estadistico_ks": float(estadistico),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }


def resumen_evidently(resultado: dict) -> dict:
    """Extrae del resultado de Evidently (`snapshot.dict()`) la fracción global y el detalle por variable."""
    fraccion, columnas = None, []
    for metrica in resultado["metrics"]:
        nombre, config = metrica["metric_name"], metrica.get("config", {})
        if nombre.startswith("DriftedColumnsCount"):
            fraccion = float(metrica["value"]["share"])
        elif nombre.startswith("ValueDrift"):
            valor, umbral = float(metrica["value"]), float(config["threshold"])
            columnas.append({"variable": config["column"], "metodo": config["method"],
                             "valor": valor, "umbral": umbral, "drift": valor >= umbral})
    if fraccion is None:
        raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")
    return {"fraccion": fraccion, "columnas": columnas}


def leer_lote(ruta: str | Path) -> tuple[pd.DataFrame, dict]:
    """Lee el lote que dejó la inferencia y sus metadatos (nivel de drift, nombre...)."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el lote {ruta}. Ejecuta primero la pipeline de inferencia.")
    meta_ruta = ruta.with_suffix(".json")
    meta = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.exists() else {"nombre": ruta.stem}
    return pd.read_csv(ruta), meta


def hay_alerta(fraccion: float, umbral: float) -> bool:
    """Alerta cuando la fracción de variables con drift supera el umbral configurado."""
    return fraccion > umbral
