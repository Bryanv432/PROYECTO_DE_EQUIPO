"""Pruebas de la lógica de drift: emulación, validación de lotes, prueba KS, Evidently y alerta.

No necesitan MLflow ni Prefect.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.config import cargar_config
from src.drift import (
    aplicar_drift, dividir_entrenamiento_prueba, drift_de_target, hay_alerta, resumen_evidently,
    tomar_lote, validar_lote, variables_afectadas,
)
from src.load_data import cargar_datos

CONFIG = cargar_config()
VARIABLES = ["marca", "tipo_vehiculo", "año_modelo", "kilometraje", "precio_lista"]


@pytest.fixture(scope="module")
def particion():
    return dividir_entrenamiento_prueba(cargar_datos(CONFIG["datos"]), 0.2, 42)


@pytest.fixture(scope="module")
def base(particion):
    return tomar_lote(particion[1], 500, 42)


# ---------- emulación ----------

def test_nivel_cero_no_cambia_el_lote(base):
    pd.testing.assert_frame_equal(aplicar_drift(base, 0.0), base)


@pytest.mark.parametrize("nivel, esperadas", [
    (0.0, []),
    (0.3, ["kilometraje"]),
    (0.6, ["kilometraje", "año_modelo", "tipo_vehiculo"]),
    (1.0, ["kilometraje", "año_modelo", "tipo_vehiculo", "marca", "precio_lista"]),
])
def test_variables_afectadas_por_nivel(nivel, esperadas):
    assert variables_afectadas(nivel) == esperadas


def test_solo_cambian_las_variables_afectadas(base):
    lote = aplicar_drift(base, 0.6)
    for variable in VARIABLES:
        cambio = not lote[variable].equals(base[variable])
        assert cambio == (variable in variables_afectadas(0.6)), variable


def test_el_precio_de_venta_sube_segun_el_nivel(base):
    for nivel in (0.3, 0.6):
        razon = aplicar_drift(base, nivel)["precio_venta"].mean() / base["precio_venta"].mean()
        assert razon == pytest.approx(1 + nivel * 0.25, rel=1e-3)


def test_la_emulacion_es_reproducible(base):
    pd.testing.assert_frame_equal(aplicar_drift(base, 0.6, semilla=7), aplicar_drift(base, 0.6, semilla=7))


@pytest.mark.parametrize("nivel", [-0.1, 1.5])
def test_nivel_fuera_de_rango(base, nivel):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        aplicar_drift(base, nivel)


# ---------- lote ----------

def test_lote_mas_grande_que_la_prueba(particion):
    with pytest.raises(ValueError):
        tomar_lote(particion[1], len(particion[1]) + 1, 42)


def test_validar_lote_sin_columna_objetivo(base):
    with pytest.raises(ValueError, match="sin etiquetas reales"):
        validar_lote(base.drop(columns="precio_venta"))


def test_validar_lote_con_etiquetas_vacias(base):
    lote = base.copy()
    lote.loc[:9, "precio_venta"] = np.nan
    with pytest.raises(ValueError, match="faltan 10 valores"):
        validar_lote(lote)


def test_validar_lote_completo_no_falla(base):
    validar_lote(base)


# ---------- drift del target (KS) ----------

def test_ks_sin_drift_con_la_misma_distribucion():
    rng = np.random.default_rng(0)
    resultado = drift_de_target(pd.Series(rng.normal(15000, 3000, 5000)), pd.Series(rng.normal(15000, 3000, 500)))
    assert not resultado["drift"]


def test_ks_detecta_precios_mas_altos():
    rng = np.random.default_rng(0)
    resultado = drift_de_target(pd.Series(rng.normal(15000, 3000, 5000)), pd.Series(rng.normal(17500, 3000, 500)))
    assert resultado["drift"] and resultado["cambio_media"] > 0.1


# ---------- Evidently y alerta ----------

def test_resumen_evidently_lee_fraccion_y_columnas():
    resultado = {"metrics": [
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 1.0, "share": 0.5}, "config": {}},
        {"metric_name": "ValueDrift(column=kilometraje)", "value": 1.2,
         "config": {"column": "kilometraje", "method": "Wasserstein distance (normed)", "threshold": 0.1}},
        {"metric_name": "ValueDrift(column=marca)", "value": 0.02,
         "config": {"column": "marca", "method": "Jensen-Shannon distance", "threshold": 0.1}},
    ]}
    resumen = resumen_evidently(resultado)
    assert resumen["fraccion"] == 0.5
    assert [c["drift"] for c in resumen["columnas"]] == [True, False]


def test_resumen_evidently_sin_metrica_global():
    with pytest.raises(ValueError):
        resumen_evidently({"metrics": []})


@pytest.mark.parametrize("fraccion, alerta", [(0.0, False), (0.2, False), (0.3, False), (0.31, True), (0.6, True)])
def test_alerta_solo_por_encima_del_umbral(fraccion, alerta):
    assert hay_alerta(fraccion, 0.3) == alerta


def test_criterio_de_exito_con_evidently(particion, base):
    """0.0 y 0.6 dan reportes distintos y solo 0.6 supera el umbral de alerta."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    from evidently import Report
    from evidently.presets import DataDriftPreset

    referencia = particion[0][VARIABLES]
    fracciones = {}
    for nivel in (0.0, 0.6):
        snapshot = Report([DataDriftPreset()]).run(reference_data=referencia,
                                                   current_data=aplicar_drift(base, nivel)[VARIABLES])
        fracciones[nivel] = resumen_evidently(snapshot.dict())["fraccion"]
    umbral = CONFIG["monitoreo"]["umbral_drift"]
    assert fracciones[0.0] != fracciones[0.6]
    assert not hay_alerta(fracciones[0.0], umbral)
    assert hay_alerta(fracciones[0.6], umbral)
