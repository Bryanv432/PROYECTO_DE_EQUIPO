"""Construcción, entrenamiento y evaluación del modelo de precio de venta."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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

MODELOS = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
}


def construir_pipeline(config_modelo: dict) -> Pipeline:
    """Preprocesado (one-hot + escalado) y regresor en un único objeto de scikit-learn."""
    tipo = config_modelo["tipo"]
    if tipo not in MODELOS:
        raise ValueError(f"Modelo no soportado: {tipo}. Opciones: {list(MODELOS)}")
    hiperparametros = config_modelo.get("hiperparametros", {}) if tipo == "random_forest" else {}

    preprocesado = ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), config_modelo["variables_categoricas"]),
            ("numericas", StandardScaler(), config_modelo["variables_numericas"]),
        ]
    )
    return Pipeline([("preprocesado", preprocesado), ("modelo", MODELOS[tipo](**hiperparametros))])


def dividir_datos(df: pd.DataFrame, config_modelo: dict):
    """Separa variables explicativas y objetivo, y divide en entrenamiento y prueba."""
    variables = config_modelo["variables_categoricas"] + config_modelo["variables_numericas"]
    X = df[variables]
    y = df[config_modelo["objetivo"]]
    return train_test_split(
        X, y, test_size=config_modelo["test_size"], random_state=config_modelo["semilla_split"]
    )


def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    """Métricas de regresión. MAE y RMSE en las mismas unidades que el precio (USD)."""
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": float(r2_score(y_real, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_real, y_pred)),
    }


def entrenar(df: pd.DataFrame, config_modelo: dict):
    """Entrena el pipeline y devuelve (pipeline, métricas, X_train, X_test)."""
    X_train, X_test, y_train, y_test = dividir_datos(df, config_modelo)
    pipeline = construir_pipeline(config_modelo)
    pipeline.fit(X_train, y_train)
    metricas = calcular_metricas(y_test, pipeline.predict(X_test))
    return pipeline, metricas, X_train, X_test
