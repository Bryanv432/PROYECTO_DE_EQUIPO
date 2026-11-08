"""Pruebas del entrenamiento del modelo."""
import pytest

from src.config import cargar_config, cargar_features
from src.features import resolver_variables
from src.generate_data import generate_car_sales_data
from src.train import calcular_metricas, construir_pipeline, dividir_datos

CONFIG = cargar_config()


@pytest.fixture(scope="module")
def datos():
    return generate_car_sales_data(num_records=15000, random_state=42, output_path=None)


def entrenar(datos, tipo, hiperparametros, seleccion="ALL"):
    variables = resolver_variables({**cargar_features(), "seleccion": seleccion})
    X_train, X_test, y_train, y_test = dividir_datos(datos, variables, "precio_venta", 0.2, 42)
    pipeline = construir_pipeline(tipo, hiperparametros, variables).fit(X_train, y_train)
    return calcular_metricas(y_test, pipeline.predict(X_test))


def test_entrenamiento_devuelve_metricas(datos):
    metricas = entrenar(datos, "random_forest", {"n_estimators": 20, "random_state": 0})
    assert set(metricas) == {"r2", "mae", "rmse", "mape"}
    assert metricas["r2"] > 0.8


def test_entrenamiento_es_reproducible(datos):
    hp = {"n_estimators": 10, "random_state": 0}
    assert entrenar(datos, "random_forest", hp) == entrenar(datos, "random_forest", hp)


@pytest.mark.parametrize("seleccion", ["ALL", "NUMERIC", "CUSTOM"])
def test_entrena_con_cada_seleccion(datos, seleccion):
    assert entrenar(datos, "linear_regression", {}, seleccion)["r2"] > 0


def test_modelo_no_soportado():
    variables = resolver_variables(cargar_features())
    with pytest.raises(ValueError):
        construir_pipeline("xgboost", {}, variables)
