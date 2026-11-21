"""Pipeline de inferencia: valora un lote de vehículos con el modelo champion.

Pasos: cargar champion -> ingesta del lote (con drift emulado) -> predecir -> guardar y registrar.

- El modelo se carga por su alias (models:/<nombre>@champion), nunca por un número de versión.
- El lote sale de la PARTE DE PRUEBA del dataset (vehículos que el modelo no vio al entrenar).
- Al lote se le aplica el drift emulado de `inferencia.nivel_drift` en configs/config.yaml
  (0.0, 0.3, 0.6...). El nivel lo fija la configuración, no quien ejecuta la pipeline.
- Se guarda una copia del lote puntuado (con el precio real) para los pipelines de drift.

Ejecución (desde la raíz del repositorio, con los servicios levantados):
    poetry run python flows/inferencia_flow.py
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.drift import aplicar_drift, dividir_entrenamiento_prueba, tomar_lote, variables_afectadas  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


def preparar_lote(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Lote de la parte de prueba con el drift emulado de la configuración. Sin Prefect ni MLflow."""
    _, prueba = dividir_entrenamiento_prueba(df, config["division"]["test_size"], config["division"]["semilla"])
    base = tomar_lote(prueba, config["inferencia"]["tamano_lote"], config["inferencia"]["semilla_lote"])
    return aplicar_drift(base, config["inferencia"]["nivel_drift"], config["datos"]["objetivo"],
                         config["inferencia"]["semilla_lote"])


def guardar_lote(lote: pd.DataFrame, nivel: float, config: dict, momento: datetime) -> dict:
    """Guarda el lote como lote_actual.csv (lo leen los pipelines de drift) y una copia con fecha y nivel."""
    carpeta = ruta_absoluta(config["monitoreo"]["carpeta_lotes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = f"lote_{momento:%Y-%m-%d_%H%M%S}_nivel_{nivel}"
    copia = carpeta / f"{nombre}.csv"
    actual = ruta_absoluta(config["monitoreo"]["lote_actual"])
    for ruta in (copia, actual):
        lote.to_csv(ruta, index=False, lineterminator="\n")
    # Metadatos del lote: los pipelines de drift los usan para nombrar los reportes.
    meta = {"nombre": nombre, "nivel_drift": nivel, "filas": len(lote), "fecha": momento.isoformat(timespec="seconds"),
            "variables_afectadas": variables_afectadas(nivel)}
    actual.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"copia": copia, "actual": actual, **meta}


def anadir_predicciones(modelo, columnas: list[str], lote: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Añade el precio predicho y el error frente al precio real. Sin Prefect ni MLflow."""
    resultado = lote.copy()
    resultado["precio_predicho"] = modelo.predict(lote[columnas]).round(2)
    resultado["error_absoluto"] = (resultado["precio_predicho"] - resultado[objetivo]).abs().round(2)
    return resultado


@task(name="cargar-champion", cache_policy=NO_CACHE)
def cargar_champion(config: dict) -> dict:
    """Carga el modelo con alias champion y las columnas con las que se entrenó."""
    logger = get_run_logger()
    nombre = config["registro"]["nombre_modelo"]
    uri = f"models:/{nombre}@champion"
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(uri)
    columnas = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    logger.info(f"Champion: {nombre} versión {version.version} | variables: {columnas}")
    return {"modelo": modelo, "version": version.version, "run_id": version.run_id, "columnas": columnas}


@task(name="ingesta-lote", cache_policy=NO_CACHE)
def ingesta_lote(config: dict) -> tuple[pd.DataFrame, dict]:
    """Toma el lote de la parte de prueba, le aplica el drift emulado y guarda una copia."""
    logger = get_run_logger()
    nivel = config["inferencia"]["nivel_drift"]
    lote = preparar_lote(cargar_datos(config["datos"]), config)
    info = guardar_lote(lote, nivel, config, datetime.now())
    afectadas = ", ".join(info["variables_afectadas"]) or "ninguna"
    logger.info(f"Lote de {len(lote)} vehículos de la parte de prueba | nivel de drift {nivel} | "
                f"variables con drift emulado: {afectadas}")
    logger.info(f"Copia del lote guardada en {info['copia'].name} (y como {info['actual'].name})")
    return lote, info


@task(name="predecir", cache_policy=NO_CACHE)
def predecir(champion: dict, lote: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Añade al lote el precio predicho y el error frente al precio real."""
    logger = get_run_logger()
    resultado = anadir_predicciones(champion["modelo"], champion["columnas"], lote, objetivo)
    logger.info(f"Precio medio predicho: {resultado['precio_predicho'].mean():,.2f} USD | "
                f"error medio del lote: {resultado['error_absoluto'].mean():,.2f} USD")
    return resultado


@task(name="guardar-y-registrar", cache_policy=NO_CACHE)
def guardar_y_registrar(predicciones: pd.DataFrame, champion: dict, info_lote: dict, config: dict) -> Path:
    """Guarda prediccion_<fecha>.csv y registra la ejecución en MLflow (tag fase = inferencia)."""
    logger = get_run_logger()
    hoy = date.today().isoformat()
    carpeta = ruta_absoluta(config["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{hoy}.csv"
    predicciones.to_csv(ruta, index=False, lineterminator="\n")
    logger.info(f"Predicciones guardadas en {ruta}")

    nivel = info_lote["nivel_drift"]
    mlflow.set_experiment(config["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"{config['mlflow']['nombre_run_inferencia']}-{hoy}-drift-{nivel}"):
        mlflow.set_tags({"fase": "inferencia", "modelo_version": champion["version"],
                         "modelo_run_id": champion["run_id"], "nivel_drift": str(nivel), "lote": info_lote["nombre"]})
        mlflow.log_params({"tamano_lote": len(predicciones), "semilla_lote": config["inferencia"]["semilla_lote"],
                           "nivel_drift": nivel, "variables_drift": ",".join(info_lote["variables_afectadas"]) or "ninguna"})
        mlflow.log_metrics({
            "filas_predichas": len(predicciones),
            "precio_medio_predicho": float(predicciones["precio_predicho"].mean()),
            "mae_lote": float(predicciones["error_absoluto"].mean()),
        })
        mlflow.log_artifact(str(ruta), artifact_path="predicciones")
        mlflow.log_artifact(str(info_lote["copia"]), artifact_path="lote")
    return ruta


@flow(name="inferencia-precio-vehiculos", log_prints=True)
def inferencia_flow() -> str:
    config = cargar_config()
    champion = cargar_champion(config)
    lote, info_lote = ingesta_lote(config)
    predicciones = predecir(champion, lote, config["datos"]["objetivo"])
    ruta = guardar_y_registrar(predicciones, champion, info_lote, config)
    print(f"Inferencia completada con la versión {champion['version']} y drift {info_lote['nivel_drift']}: "
          f"{info_lote['nombre']}.csv")
    return str(ruta)


if __name__ == "__main__":
    inferencia_flow()
