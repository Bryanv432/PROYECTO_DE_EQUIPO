"""Pipeline de entrenamiento del equipo: precio de venta de vehículos usados.

Tareas: cargar datos -> entrenar y registrar en MLflow -> resumen (artefacto en Prefect).

Requisitos: servicios levantados (services/docker-compose.yml) y, en la misma terminal:
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Ejecución (desde la raíz del repositorio):
    poetry run python flows/entrenamiento_flow.py

Resultado:
    - Prefect (http://localhost:4200/runs): run del flow "entrenamiento-precio-vehiculos"
      en estado Completed, con logs del tamaño del dataset y el artefacto "resumen-entrenamiento".
    - MLflow (http://localhost:5000): experimento "precio-vehiculos-a1" con parámetros,
      métricas (MAE, RMSE, R2, MAPE), dataset de entrada, config.yaml y el modelo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import RUTA_CONFIG, cargar_config, ruta_absoluta  # noqa: E402
from src.load_data import cargar_datos, huella_archivo  # noqa: E402
from src.train import entrenar  # noqa: E402


@task(name="cargar-datos", cache_policy=NO_CACHE)
def tarea_cargar_datos(config_datos: dict) -> pd.DataFrame:
    logger = get_run_logger()
    df = cargar_datos(config_datos)
    logger.info(f"Dataset cargado: {len(df)} filas x {df.shape[1]} columnas")
    logger.info(f"Archivo: {config_datos['ruta']} | MD5: {huella_archivo(ruta_absoluta(config_datos['ruta']))}")
    logger.info(f"Rango de fechas: {df['fecha_venta'].min().date()} a {df['fecha_venta'].max().date()}")
    return df


@task(name="entrenar-y-registrar", cache_policy=NO_CACHE)
def tarea_entrenar(df: pd.DataFrame, config: dict) -> dict:
    logger = get_run_logger()
    config_datos, config_modelo, config_mlflow = config["datos"], config["modelo"], config["mlflow"]

    mlflow.set_experiment(config_mlflow["experimento"])
    with mlflow.start_run(run_name=config_mlflow["nombre_run"]) as run:
        pipeline, metricas, X_train, X_test = entrenar(df, config_modelo)

        # Parámetros: datos y modelo
        mlflow.log_params(
            {
                "n_filas": len(df),
                "n_columnas": df.shape[1],
                "semilla_datos": config_datos["semilla"],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_size": config_modelo["test_size"],
                "tipo_modelo": config_modelo["tipo"],
                "variables": ",".join(X_train.columns),
                **{f"hp_{k}": v for k, v in config_modelo.get("hiperparametros", {}).items()},
            }
        )
        mlflow.log_metrics(metricas)

        # Trazabilidad: versión exacta de los datos y de la configuración
        ruta_datos = ruta_absoluta(config_datos["ruta"])
        mlflow.set_tag("dataset_md5", huella_archivo(ruta_datos))
        mlflow.log_input(
            mlflow.data.from_pandas(df, source=str(config_datos["ruta"]), name="car_sales_dataset",
                                    targets=config_modelo["objetivo"]),
            context="training",
        )
        mlflow.log_artifact(str(RUTA_CONFIG), artifact_path="config")

        # Modelo con firma y ejemplo de entrada
        ejemplo = X_test.head(5)
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            signature=infer_signature(ejemplo, pipeline.predict(ejemplo)),
            input_example=ejemplo,
            # MLflow 3.17 serializa con skops: los árboles del Random Forest deben declararse
            # como tipo de confianza. Es seguro porque el modelo lo entrena este mismo pipeline.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.info("Métricas en prueba: " + ", ".join(f"{k}={v:,.4f}" for k, v in metricas.items()))
        logger.info(f"Run de MLflow: {run.info.run_id}")
        return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id, "metricas": metricas,
                "n_train": len(X_train), "n_test": len(X_test)}


@task(name="resumen", cache_policy=NO_CACHE)
def tarea_resumen(resultado: dict, df: pd.DataFrame, config: dict) -> None:
    m = resultado["metricas"]
    markdown = f"""# Resumen del entrenamiento

| Concepto | Valor |
|---|---|
| Modelo | `{config['modelo']['tipo']}` |
| Filas del dataset | {len(df):,} |
| Entrenamiento / prueba | {resultado['n_train']:,} / {resultado['n_test']:,} |
| MAE (USD) | {m['mae']:,.2f} |
| RMSE (USD) | {m['rmse']:,.2f} |
| R² | {m['r2']:.4f} |
| MAPE | {m['mape']:.2%} |
| Experimento MLflow | `{config['mlflow']['experimento']}` |
| Run ID MLflow | `{resultado['run_id']}` |
"""
    create_markdown_artifact(key="resumen-entrenamiento", markdown=markdown,
                             description="Resumen del entrenamiento del modelo de precio de venta")


@flow(name="entrenamiento-precio-vehiculos", log_prints=True)
def entrenamiento_flow() -> dict:
    config = cargar_config()
    df = tarea_cargar_datos(config["datos"])
    resultado = tarea_entrenar(df, config)
    tarea_resumen(resultado, df, config)
    print(f"Entrenamiento completado. R² = {resultado['metricas']['r2']:.4f}")
    return resultado


if __name__ == "__main__":
    entrenamiento_flow()
