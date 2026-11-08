"""Lógica del modelo: pipeline de scikit-learn, división de datos y métricas.

No usa Prefect ni MLflow, así que se puede probar sin servicios levantados.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import Variables

MODELOS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "linear_regression": LinearRegression,
}


def construir_pipeline(tipo: str, hiperparametros: dict, variables: Variables) -> Pipeline:
    """Preprocesado (one-hot y escalado) y regresor en un único objeto de scikit-learn."""
    if tipo not in MODELOS:
        raise ValueError(f"Modelo no soportado: {tipo}. Opciones: {list(MODELOS)}")
    transformaciones = []
    if variables.categoricas:
        transformaciones.append(("categoricas", OneHotEncoder(handle_unknown="ignore"), variables.categoricas))
    if variables.numericas:
        transformaciones.append(("numericas", StandardScaler(), variables.numericas))
    return Pipeline([
        ("preprocesado", ColumnTransformer(transformaciones)),
        ("modelo", MODELOS[tipo](**(hiperparametros or {}))),
    ])


def dividir_datos(df: pd.DataFrame, variables: Variables, objetivo: str, test_size: float, semilla: int):
    """Separa variables y objetivo, y divide en entrenamiento y prueba."""
    return train_test_split(df[variables.todas], df[objetivo], test_size=test_size, random_state=semilla)


def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    """Métricas de regresión. MAE y RMSE en USD; R² entre 0 y 1 (más alto es mejor)."""
    return {
        "r2": float(r2_score(y_real, y_pred)),
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_real, y_pred)),
    }
