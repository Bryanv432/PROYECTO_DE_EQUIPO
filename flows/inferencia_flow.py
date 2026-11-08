"""Pipeline de inferencia: valora un lote de vehículos con el modelo champion.

Pasos: cargar champion -> tomar lote -> predecir -> guardar y registrar.

El modelo se carga por su alias (models:/<nombre>@champion), nunca por un número de versión:
si el entrenamiento promueve un modelo nuevo, la siguiente inferencia lo usa sin tocar código.

Ejecución (desde la raíz del repositorio, con los servicios levantados):
    poetry run python flows/inferencia_flow.py
"""
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


@task(name="cargar-champion", cache_policy=NO_CACHE)
def cargar_champion(config: dict) -> dict:
    """Carga el modelo con alias champion y las columnas con las que se entrenó."""
    logger = get_run_logger()
    nombre = config["registro"]["nombre_modelo"]
    uri = f"models:/{nombre}@champion"
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(uri)
    # Las columnas salen de la firma del modelo: la inferencia usa las mismas del entrenamiento,
    # aunque features.yaml haya cambiado después.
    columnas = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    logger.info(f"Champion: {nombre} versión {version.version} | variables: {columnas}")
    return {"modelo": modelo, "version": version.version, "run_id": version.run_id, "columnas": columnas}


@task(name="tomar-lote", cache_policy=NO_CACHE)
def tomar_lote(config: dict) -> tuple[pd.DataFrame, int]:
    """Toma una muestra al azar del dataset para simular los vehículos que llegan en el día."""
    logger = get_run_logger()
    semilla = random.randint(0, 1_000_000)  # cambia en cada ejecución; se registra para poder repetirla
    df = cargar_datos(config["datos"])
    lote = df.sample(n=config["inferencia"]["tamano_lote"], random_state=semilla).reset_index(drop=True)
    logger.info(f"Lote de {len(lote)} vehículos (semilla {semilla})")
    return lote, semilla


@task(name="predecir", cache_policy=NO_CACHE)
def predecir(champion: dict, lote: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Añade al lote el precio predicho y el error frente al precio real."""
    logger = get_run_logger()
    resultado = lote.copy()
    resultado["precio_predicho"] = champion["modelo"].predict(lote[champion["columnas"]]).round(2)
    resultado["error_absoluto"] = (resultado["precio_predicho"] - resultado[objetivo]).abs().round(2)
    logger.info(f"Precio medio predicho: {resultado['precio_predicho'].mean():,.2f} USD | "
                f"error medio del lote: {resultado['error_absoluto'].mean():,.2f} USD")
    return resultado


@task(name="guardar-y-registrar", cache_policy=NO_CACHE)
def guardar_y_registrar(predicciones: pd.DataFrame, champion: dict, semilla: int, config: dict) -> Path:
    """Guarda prediccion_<fecha>.csv y registra la ejecución en MLflow (tag fase = inferencia)."""
    logger = get_run_logger()
    hoy = date.today().isoformat()
    carpeta = ruta_absoluta(config["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{hoy}.csv"
    predicciones.to_csv(ruta, index=False, lineterminator="\n")
    logger.info(f"Predicciones guardadas en {ruta}")

    mlflow.set_experiment(config["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"{config['mlflow']['nombre_run_inferencia']}-{hoy}"):
        mlflow.set_tags({"fase": "inferencia", "modelo_version": champion["version"],
                         "modelo_run_id": champion["run_id"]})
        mlflow.log_params({"tamano_lote": len(predicciones), "semilla_lote": semilla})
        mlflow.log_metrics({
            "filas_predichas": len(predicciones),
            "precio_medio_predicho": float(predicciones["precio_predicho"].mean()),
            "mae_lote": float(predicciones["error_absoluto"].mean()),
        })
        mlflow.log_artifact(str(ruta), artifact_path="predicciones")
    return ruta


@flow(name="inferencia-precio-vehiculos", log_prints=True)
def inferencia_flow() -> str:
    config = cargar_config()
    champion = cargar_champion(config)
    lote, semilla = tomar_lote(config)
    predicciones = predecir(champion, lote, config["datos"]["objetivo"])
    ruta = guardar_y_registrar(predicciones, champion, semilla, config)
    print(f"Inferencia completada con la versión {champion['version']}: {ruta.name}")
    return str(ruta)


if __name__ == "__main__":
    inferencia_flow()
