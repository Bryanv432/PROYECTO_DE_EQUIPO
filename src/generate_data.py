"""Generación del dataset sintético de ventas de vehículos usados.

Uso (desde la raíz del repositorio):
    poetry run python src/generate_data.py

Los parámetros (número de filas, semilla y ruta de salida) se leen de configs/config.yaml.
Con la misma semilla, el CSV generado es idéntico en cada ejecución.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite `python src/generate_data.py`

from src.config import cargar_config, ruta_absoluta  # noqa: E402


def generate_car_sales_data(
    num_records: int = 20000,
    random_state: int = 42,
    output_path: str | Path | None = "data/car_sales_dataset.csv",
) -> pd.DataFrame:
    """
    Genera un dataset sintético reproducible de ventas de autos multimarca.

    Parameters
    ----------
    num_records : int
        Número de filas a generar (entre 15.000 y 30.000). Por defecto 20.000.
    random_state : int
        Semilla que garantiza la reproducibilidad exacta. Por defecto 42.
    output_path : str | Path | None
        Ruta del CSV (relativa a la raíz del repositorio o absoluta).
        Si es None, no se guarda el archivo.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos generados.
    """
    if not 15000 <= num_records <= 30000:
        raise ValueError(f"num_records debe estar entre 15.000 y 30.000 (recibido: {num_records})")

    # 1. Fijar semilla aleatoria para reproducibilidad
    np.random.seed(random_state)

    # 2. Definición de catálogos y probabilidades
    marcas = ["Toyota", "Chevrolet", "Renault", "Mazda", "Nissan", "Kia", "Ford"]
    prob_marcas = [0.20, 0.18, 0.18, 0.15, 0.12, 0.10, 0.07]

    tipos_vehiculo = ["Sedán", "SUV", "Hatchback", "Pick-up"]
    prob_tipos = [0.35, 0.40, 0.15, 0.10]

    # Precios base promedio aproximados por tipo de vehículo (en USD)
    base_prices = {
        "Hatchback": 16000,
        "Sedán": 21000,
        "SUV": 31000,
        "Pick-up": 38000,
    }

    # Factor de ajuste de precio relativo por marca
    brand_multiplier = {
        "Toyota": 1.15,
        "Mazda": 1.10,
        "Ford": 1.05,
        "Chevrolet": 1.00,
        "Nissan": 0.98,
        "Renault": 0.92,
        "Kia": 0.95,
    }

    # 3. Generación de columnas
    # Fechas de venta distribuidas en un rango de 3 años (2023 a 2025)
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    random_days = np.random.randint(0, (end_date - start_date).days, size=num_records)
    fechas_venta = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

    # Marca y tipo de vehículo
    marcas_gen = np.random.choice(marcas, size=num_records, p=prob_marcas)
    tipos_gen = np.random.choice(tipos_vehiculo, size=num_records, p=prob_tipos)

    # Año modelo del vehículo (2015 - 2025)
    años_modelo = np.random.randint(2015, 2026, size=num_records)

    # Kilometraje: directamente correlacionado con la antigüedad del auto
    # Autos más antiguos tienen más kilometraje promedio por año (12,000 - 18,000 km/año)
    años_antiguedad = 2026 - años_modelo
    kilometraje = (
        años_antiguedad * np.random.uniform(11000, 17000, size=num_records)
        + np.random.normal(0, 3000, size=num_records)
    )
    kilometraje = np.clip(kilometraje, 0, None).round(0).astype(int)

    # Precio de lista sugerido (basado en tipo, marca y depreciación por año/km)
    precios_lista = []
    for tipo, marca, anio, km in zip(tipos_gen, marcas_gen, años_modelo, kilometraje):
        p_base = base_prices[tipo] * brand_multiplier[marca]
        # Depreciación acumulada (~7% por año de antigüedad)
        depreciacion_anio = (1 - 0.07) ** (2026 - anio)
        # Descuento adicional por alto kilometraje
        depreciacion_km = np.maximum(0.60, 1.0 - (km / 300000) * 0.40)
        p_calculado = p_base * depreciacion_anio * depreciacion_km
        # Ruido de variación de mercado
        p_final = p_calculado * np.random.uniform(0.95, 1.05)
        precios_lista.append(round(p_final, 2))

    precios_lista = np.array(precios_lista)

    # Precio final de venta (Target): varía según negociación/temporada
    # Margen o descuento de negociación entre -8% y +2%
    descuento_negociacion = np.random.uniform(0.92, 1.02, size=num_records)
    precios_venta = np.round(precios_lista * descuento_negociacion, 2)

    # 4. Construcción del DataFrame
    df = pd.DataFrame(
        {
            "fecha_venta": fechas_venta,
            "marca": marcas_gen,
            "tipo_vehiculo": tipos_gen,
            "año_modelo": años_modelo,
            "kilometraje": kilometraje,
            "precio_lista": precios_lista,
            "precio_venta": precios_venta,
        }
    )

    # Ordenar cronológicamente por fecha
    df = df.sort_values(by="fecha_venta").reset_index(drop=True)

    # 5. Guardado asegurando la existencia de la ruta
    if output_path is not None:
        target_path = ruta_absoluta(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # lineterminator fijo: el CSV (y su MD5) es idéntico en Windows, Linux y macOS
        df.to_csv(target_path, index=False, lineterminator="\n")
        print(f"Dataset generado con {len(df)} registros y {len(df.columns)} columnas.")
        print(f"Ubicación: {target_path}")

    return df


if __name__ == "__main__":
    config = cargar_config()["datos"]
    generate_car_sales_data(
        num_records=config["num_registros"],
        random_state=config["semilla"],
        output_path=config["ruta"],
    )
