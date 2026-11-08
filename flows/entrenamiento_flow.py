"""Pipeline de entrenamiento: precio de venta de vehículos usados.

Pasos: ingesta -> entrenamiento -> evaluación -> registro.

En el registro, el modelo nuevo se compara con el champion actual de MLflow. Solo se registra
como versión nueva y recibe el alias `champion` si cumple las reglas de `registro` en
configs/config.yaml. Si no, el run queda en MLflow pero el alias no cambia.

Ejecución (desde la raíz del repositorio, con los servicios levantados):
    $env:PREFECT_API_URL = "http://localhost:4200/api"
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    poetry run python flows/entrenamiento_flow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.champion import decidir_promocion  # noqa: E402
from src.config import RUTA_CONFIG, RUTA_FEATURES, cargar_config, cargar_features, ruta_absoluta  # noqa: E402
from src.features import Variables, resolver_variables  # noqa: E402
from src.load_data import cargar_datos, huella_archivo  # noqa: E402
from src.train import calcular_metricas, construir_pipeline, dividir_datos  # noqa: E402

# MLflow 3.17 guarda los modelos con skops y pide declarar los tipos de los árboles.
# Es seguro: el modelo lo entrena esta misma pipeline.
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree"]


@task(name="ingesta", cache_policy=NO_CACHE)
def ingesta(config: dict, config_features: dict) -> tuple[pd.DataFrame, Variables]:
    """Carga y valida el dataset, y resuelve las variables del experimento."""
    logger = get_run_logger()
    df = cargar_datos(config["datos"])
    variables = resolver_variables(config_features, list(df.columns))
    logger.info(f"Dataset: {len(df)} filas x {df.shape[1]} columnas | "
                f"MD5: {huella_archivo(ruta_absoluta(config['datos']['ruta']))}")
    logger.info(f"Variables ({variables.alias}, {variables.n}): {variables.todas}")
    return df, variables


@task(name="entrenamiento", cache_policy=NO_CACHE)
def entrenamiento(df: pd.DataFrame, variables: Variables, config: dict):
    """Divide los datos y entrena el modelo indicado en config.yaml."""
    logger = get_run_logger()
    tipo = config["modelo"]["tipo"]
    hiperparametros = config["modelo"]["hiperparametros"].get(tipo, {})
    X_train, X_test, y_train, y_test = dividir_datos(
        df, variables, config["datos"]["objetivo"], config["division"]["test_size"], config["division"]["semilla"])
    logger.info(f"Entrenando {tipo} con {hiperparametros} | train={len(X_train)} test={len(X_test)}")
    pipeline = construir_pipeline(tipo, hiperparametros, variables)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_test


@task(name="evaluacion", cache_policy=NO_CACHE)
def evaluacion(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcula las métricas sobre el conjunto de prueba."""
    logger = get_run_logger()
    metricas = calcular_metricas(y_test, pipeline.predict(X_test))
    logger.info("Métricas en prueba: " + ", ".join(f"{k}={v:,.4f}" for k, v in metricas.items()))
    return metricas


def metrica_del_champion(client: MlflowClient, nombre_modelo: str, metrica: str):
    """Devuelve (versión, valor de la métrica) del champion vigente, o (None, None) si no hay."""
    try:
        version = client.get_model_version_by_alias(nombre_modelo, "champion")
    except MlflowException:
        return None, None
    return version.version, client.get_run(version.run_id).data.metrics[metrica]


@task(name="registro", cache_policy=NO_CACHE)
def registro(pipeline, metricas: dict, df: pd.DataFrame, variables: Variables,
             X_train: pd.DataFrame, X_test: pd.DataFrame, config: dict) -> dict:
    """Registra el run en MLflow y promueve el modelo a champion solo si mejora."""
    logger = get_run_logger()
    cfg_reg, cfg_ml = config["registro"], config["mlflow"]
    tipo = config["modelo"]["tipo"]
    client = MlflowClient()

    mlflow.set_experiment(cfg_ml["experimento"])
    nombre_run = f"{cfg_ml['nombre_run_entrenamiento']}-{tipo}-{variables.alias.lower()}"
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tags({"fase": "entrenamiento",
                         "dataset_md5": huella_archivo(ruta_absoluta(config["datos"]["ruta"]))})
        mlflow.log_params({
            "tipo_modelo": tipo,
            "features": variables.alias,
            "n_features": variables.n,
            "variables": ",".join(variables.todas),
            "n_filas": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": config["division"]["test_size"],
            "semilla": config["division"]["semilla"],
            **{f"hp_{k}": v for k, v in config["modelo"]["hiperparametros"].get(tipo, {}).items()},
        })
        mlflow.log_metrics(metricas)
        mlflow.log_input(mlflow.data.from_pandas(df, source=str(config["datos"]["ruta"]),
                                                 name="car_sales_dataset", targets=config["datos"]["objetivo"]),
                         context="training")
        mlflow.log_artifact(str(RUTA_CONFIG), artifact_path="config")
        mlflow.log_artifact(str(RUTA_FEATURES), artifact_path="config")
        ejemplo = X_test.head(5)
        info = mlflow.sklearn.log_model(pipeline, name="modelo",
                                        signature=infer_signature(ejemplo, pipeline.predict(ejemplo)),
                                        input_example=ejemplo, skops_trusted_types=TIPOS_CONFIABLES)

        version_actual, valor_champion = metrica_del_champion(client, cfg_reg["nombre_modelo"], cfg_reg["metrica"])
        logger.info(f"Champion actual: {'ninguno' if version_actual is None else f'versión {version_actual}'}"
                    + ("" if valor_champion is None else f" ({cfg_reg['metrica']}={valor_champion:.4f})"))
        decision = decidir_promocion(metricas, valor_champion, cfg_reg["metrica"], cfg_reg["umbral_minimo"],
                                     cfg_reg.get("mejora_minima", 0.0), cfg_reg.get("mae_maximo"))
        mlflow.set_tags({"promovido": str(decision.promover).lower(), "motivo": decision.motivo})

        if not decision.promover:
            mensaje = f"No se promueve: {decision.motivo}. El alias champion no cambia."
            logger.info(mensaje)
            return {"run_id": run.info.run_id, "promovido": False, "mensaje": mensaje}

        version = mlflow.register_model(info.model_uri, cfg_reg["nombre_modelo"])
        client.set_registered_model_alias(cfg_reg["nombre_modelo"], "champion", version.version)
        client.set_model_version_tag(cfg_reg["nombre_modelo"], version.version, "estado", "aprobado")
        client.set_model_version_tag(cfg_reg["nombre_modelo"], version.version, "features", variables.alias)
        mlflow.set_tag("version_registrada", version.version)
        mensaje = f"Versión {version.version} registrada como champion ({decision.motivo})"
        logger.info(mensaje)
        return {"run_id": run.info.run_id, "promovido": True, "version": version.version, "mensaje": mensaje}


@flow(name="entrenamiento-precio-vehiculos", log_prints=True)
def entrenamiento_flow() -> dict:
    config, config_features = cargar_config(), cargar_features()
    df, variables = ingesta(config, config_features)
    pipeline, X_train, X_test, y_test = entrenamiento(df, variables, config)
    metricas = evaluacion(pipeline, X_test, y_test)
    resultado = registro(pipeline, metricas, df, variables, X_train, X_test, config)
    print(resultado["mensaje"])
    return resultado


if __name__ == "__main__":
    entrenamiento_flow()
