"""Pruebas de la inferencia sin MLflow ni Prefect: lote de prueba, drift, copia del lote y predicción."""
import json
from datetime import datetime

import pandas as pd
import pytest

from flows.inferencia_flow import anadir_predicciones, guardar_lote, preparar_lote
from src.config import cargar_config
from src.drift import dividir_entrenamiento_prueba, leer_lote
from src.load_data import cargar_datos

CONFIG = cargar_config()


@pytest.fixture(scope="module")
def datos():
    return cargar_datos(CONFIG["datos"])


def config_con(nivel, tmp_path=None):
    config = {**CONFIG, "inferencia": {**CONFIG["inferencia"], "nivel_drift": nivel}}
    if tmp_path is not None:
        config["monitoreo"] = {**CONFIG["monitoreo"], "carpeta_lotes": str(tmp_path),
                               "lote_actual": str(tmp_path / "lote_actual.csv")}
    return config


def test_el_lote_sale_solo_de_la_parte_de_prueba(datos):
    lote = preparar_lote(datos, config_con(0.0))
    entrenamiento, prueba = dividir_entrenamiento_prueba(datos, CONFIG["division"]["test_size"],
                                                         CONFIG["division"]["semilla"])
    claves = list(lote.columns)
    assert len(lote.merge(prueba, on=claves)) == len(lote)
    assert lote.merge(entrenamiento, on=claves).empty


def test_el_lote_tiene_el_tamano_configurado(datos):
    assert len(preparar_lote(datos, config_con(0.0))) == CONFIG["inferencia"]["tamano_lote"]


def test_el_nivel_de_la_configuracion_se_aplica(datos):
    sin_drift, con_drift = preparar_lote(datos, config_con(0.0)), preparar_lote(datos, config_con(0.6))
    assert con_drift["kilometraje"].mean() > sin_drift["kilometraje"].mean()
    assert con_drift["precio_venta"].mean() > sin_drift["precio_venta"].mean()


def test_guardar_lote_deja_copia_lote_actual_y_metadatos(datos, tmp_path):
    config = config_con(0.6, tmp_path)
    lote = preparar_lote(datos, config)
    info = guardar_lote(lote, 0.6, config, datetime(2026, 11, 20, 8, 30, 0))
    assert info["copia"].name == "lote_2026-11-20_083000_nivel_0.6.csv"
    leido, meta = leer_lote(info["actual"])
    assert len(leido) == len(lote) and meta["nivel_drift"] == 0.6
    assert json.loads(info["actual"].with_suffix(".json").read_text(encoding="utf-8"))["variables_afectadas"] == [
        "kilometraje", "año_modelo", "tipo_vehiculo"]


def test_leer_lote_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="inferencia"):
        leer_lote(tmp_path / "no_existe.csv")


class ModeloFalso:
    def predict(self, X):
        return X["precio_lista"].to_numpy() * 0.95


def test_anadir_predicciones_calcula_precio_y_error():
    lote = pd.DataFrame({"precio_lista": [20000.0, 10000.0], "precio_venta": [19000.0, 9000.0]})
    resultado = anadir_predicciones(ModeloFalso(), ["precio_lista"], lote, "precio_venta")
    assert resultado["precio_predicho"].tolist() == [19000.0, 9500.0]
    assert resultado["error_absoluto"].tolist() == [0.0, 500.0]
