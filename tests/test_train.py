"""Pruebas del entrenamiento del modelo."""
import pytest

from src.config import cargar_config
from src.generate_data import generate_car_sales_data
from src.train import construir_pipeline, entrenar


@pytest.fixture(scope="module")
def datos():
    return generate_car_sales_data(num_records=15000, random_state=42, output_path=None)


def test_entrenamiento_devuelve_metricas(datos):
    config_modelo = {**cargar_config()["modelo"], "hiperparametros": {"n_estimators": 20, "random_state": 0}}
    _, metricas, _, _ = entrenar(datos, config_modelo)
    assert set(metricas) == {"mae", "rmse", "r2", "mape"}
    assert metricas["r2"] > 0.8  # el modelo debe superar ampliamente a predecir la media


def test_entrenamiento_es_reproducible(datos):
    config_modelo = {**cargar_config()["modelo"], "hiperparametros": {"n_estimators": 10, "random_state": 0}}
    _, m1, _, _ = entrenar(datos, config_modelo)
    _, m2, _, _ = entrenar(datos, config_modelo)
    assert m1 == m2


def test_modelo_no_soportado():
    with pytest.raises(ValueError):
        construir_pipeline({**cargar_config()["modelo"], "tipo": "xgboost"})
