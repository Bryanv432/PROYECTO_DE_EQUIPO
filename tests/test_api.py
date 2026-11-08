"""Pruebas de la API sin MLflow: se reemplaza la carga del champion por un modelo de prueba."""
import pytest
from fastapi.testclient import TestClient

import api.main as api_main

VEHICULO = {"marca": "Toyota", "tipo_vehiculo": "SUV", "año_modelo": 2020, "kilometraje": 60000, "precio_lista": 22000.0}


class ModeloFalso:
    """Predice el 95 % del precio de lista, para comprobar el flujo de la API."""
    def predict(self, X):
        return X["precio_lista"].to_numpy() * 0.95


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(api_main, "cargar_champion", lambda: {
        "modelo": ModeloFalso(), "version": "7", "features": "ALL",
        "columnas": ["marca", "tipo_vehiculo", "año_modelo", "kilometraje", "precio_lista"]})
    with TestClient(api_main.app) as c:
        yield c


def test_health_muestra_la_version(cliente):
    assert cliente.get("/health").json()["version"] == "7"


def test_predict(cliente):
    respuesta = cliente.post("/predict", json=VEHICULO).json()
    assert respuesta == {"precio_predicho": 20900.0, "version_modelo": "7", "features": "ALL"}


def test_predict_valida_los_datos(cliente):
    assert cliente.post("/predict", json={**VEHICULO, "kilometraje": -5}).status_code == 422
