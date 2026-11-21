"""Pruebas de reproducibilidad y validación de los datos."""
import pandas as pd
import pytest

from src.config import cargar_config
from src.generate_data import generate_car_sales_data
from src.load_data import cargar_datos, validar_columnas

COLUMNAS = cargar_config()["datos"]["columnas_esperadas"]


def test_misma_semilla_mismo_dataset():
    a = generate_car_sales_data(num_records=15000, random_state=42, output_path=None)
    b = generate_car_sales_data(num_records=15000, random_state=42, output_path=None)
    pd.testing.assert_frame_equal(a, b)


def test_distinta_semilla_distinto_dataset():
    a = generate_car_sales_data(num_records=15000, random_state=1, output_path=None)
    b = generate_car_sales_data(num_records=15000, random_state=2, output_path=None)
    assert not a.equals(b)


def test_estructura_y_rangos():
    df = generate_car_sales_data(num_records=15000, random_state=42, output_path=None)
    assert list(df.columns) == COLUMNAS
    assert len(df) == 15000
    assert df.notna().all().all()
    assert df["año_modelo"].between(2015, 2025).all()
    assert (df["kilometraje"] >= 0).all()
    assert (df["precio_venta"] > 0).all()


def test_num_registros_fuera_de_rango():
    with pytest.raises(ValueError):
        generate_car_sales_data(num_records=100, output_path=None)


def test_validar_columnas_detecta_faltantes():
    with pytest.raises(ValueError, match="precio_venta"):
        validar_columnas(pd.DataFrame({"marca": ["Kia"]}), ["marca", "precio_venta"])


def test_cargar_datos_regenera_si_no_existe(tmp_path):
    config = {**cargar_config()["datos"], "ruta": tmp_path / "datos.csv", "num_registros": 15000}
    df = cargar_datos(config)
    assert (tmp_path / "datos.csv").exists()
    assert len(df) == 15000


# ---------- Actividad 3: tests de datos sobre el CSV versionado (esquema, rangos y nulos) ----------

MARCAS = {"Toyota", "Chevrolet", "Renault", "Mazda", "Nissan", "Kia", "Ford"}
TIPOS = {"Sedán", "SUV", "Hatchback", "Pick-up"}


@pytest.fixture(scope="module")
def dataset():
    return cargar_datos(cargar_config()["datos"])


def test_esquema_columnas_y_tipos(dataset):
    assert list(dataset.columns) == COLUMNAS
    assert pd.api.types.is_datetime64_any_dtype(dataset["fecha_venta"])
    for columna in ("año_modelo", "kilometraje"):
        assert pd.api.types.is_integer_dtype(dataset[columna]), columna
    for columna in ("precio_lista", "precio_venta"):
        assert pd.api.types.is_float_dtype(dataset[columna]), columna


def test_sin_nulos_ni_duplicados(dataset):
    assert dataset.notna().all().all()
    assert not dataset.duplicated().any()


def test_rangos_de_las_variables(dataset):
    assert dataset["año_modelo"].between(2015, 2025).all()
    assert dataset["kilometraje"].between(0, 400_000).all()
    assert (dataset["precio_lista"] > 0).all() and (dataset["precio_venta"] > 0).all()
    assert dataset["fecha_venta"].between("2023-01-01", "2025-12-31").all()


def test_categorias_validas(dataset):
    assert set(dataset["marca"]) <= MARCAS
    assert set(dataset["tipo_vehiculo"]) <= TIPOS


def test_precio_de_venta_coherente_con_el_de_lista(dataset):
    # La negociación va de -8 % a +2 % sobre el precio de lista.
    razon = dataset["precio_venta"] / dataset["precio_lista"]
    assert razon.between(0.919, 1.021).all()
