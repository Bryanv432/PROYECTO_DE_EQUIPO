"""Pruebas de la regla que decide quién es champion."""
from src.champion import decidir_promocion

REGLAS = {"metrica": "r2", "umbral_minimo": 0.90, "mejora_minima": 0.001, "mae_maximo": 1000.0}


def test_primer_modelo_que_supera_el_minimo_es_champion():
    assert decidir_promocion({"r2": 0.95, "mae": 500}, None, **REGLAS).promover


def test_no_alcanza_el_minimo():
    assert not decidir_promocion({"r2": 0.85, "mae": 500}, None, **REGLAS).promover


def test_error_demasiado_alto():
    assert not decidir_promocion({"r2": 0.95, "mae": 1500}, None, **REGLAS).promover


def test_mejora_al_champion():
    assert decidir_promocion({"r2": 0.99, "mae": 400}, 0.98, **REGLAS).promover


def test_no_mejora_lo_suficiente():
    # Mejora en 0,0005, menos que la mejora mínima de 0,001
    assert not decidir_promocion({"r2": 0.9805, "mae": 400}, 0.98, **REGLAS).promover


def test_empate_no_cambia_el_champion():
    assert not decidir_promocion({"r2": 0.98, "mae": 400}, 0.98, **REGLAS).promover
