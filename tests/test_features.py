"""Pruebas de la selección de variables (configs/features.yaml)."""
import pytest

from src.features import resolver_variables

BASE = {
    "numericas": ["año_modelo", "kilometraje", "precio_lista"],
    "categoricas": ["marca", "tipo_vehiculo"],
    "custom": ["marca", "año_modelo"],
}


def test_all_usa_todas_las_variables():
    v = resolver_variables({**BASE, "seleccion": "ALL"})
    assert v.n == 5 and v.categoricas == ["marca", "tipo_vehiculo"]


def test_numeric_usa_solo_numericas():
    v = resolver_variables({**BASE, "seleccion": "NUMERIC"})
    assert v.categoricas == [] and v.numericas == BASE["numericas"]


def test_custom_separa_por_tipo():
    v = resolver_variables({**BASE, "seleccion": "custom"})  # no distingue mayúsculas
    assert v.alias == "CUSTOM" and v.categoricas == ["marca"] and v.numericas == ["año_modelo"]


def test_custom_con_variable_desconocida():
    with pytest.raises(ValueError, match="color"):
        resolver_variables({**BASE, "seleccion": "CUSTOM", "custom": ["color"]})


def test_seleccion_no_valida():
    with pytest.raises(ValueError):
        resolver_variables({**BASE, "seleccion": "TODAS"})


def test_detecta_variable_ausente_en_dataset():
    with pytest.raises(ValueError, match="precio_lista"):
        resolver_variables({**BASE, "seleccion": "ALL"}, ["marca", "tipo_vehiculo", "año_modelo", "kilometraje"])
