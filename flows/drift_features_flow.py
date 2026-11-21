"""Pipeline de drift de features: compara cada variable de entrada del lote con el entrenamiento.

Pasos: referencia -> lote -> reporte de Evidently -> alerta -> registro.

- Referencia: la parte de ENTRENAMIENTO del dataset (las mismas filas con las que se entrena).
- Lote: el último que puntuó la inferencia (`monitoreo.lote_actual`).
- Evidently compara la distribución de las 5 variables y guarda un reporte HTML por lote en `reports/`.
- Si la fracción de variables con drift supera `monitoreo.umbral_drift`, emite una ALERTA (log de nivel WARNING).
- Si MLFLOW_TRACKING_URI está definida, también registra el resultado en MLflow (tag fase = monitoreo).

Ejecución (desde la raíz del repositorio, después de la inferencia):
    poetry run python flows/drift_features_flow.py
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/
warnings.filterwarnings("ignore", category=SyntaxWarning)  # aviso interno de Evidently, no afecta

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, cargar_features, ruta_absoluta  # noqa: E402
from src.drift import dividir_entrenamiento_prueba, hay_alerta, leer_lote, resumen_evidently  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


def variables_de_entrada() -> list[str]:
    """Las 5 variables de entrada del problema, sin importar con cuáles se entrenó el champion."""
    cfg = cargar_features()
    return cfg["categoricas"] + cfg["numericas"]


@task(name="referencia", cache_policy=NO_CACHE)
def referencia(config: dict) -> pd.DataFrame:
    entrenamiento, _ = dividir_entrenamiento_prueba(cargar_datos(config["datos"]),
                                                    config["division"]["test_size"], config["division"]["semilla"])
    get_run_logger().info(f"Referencia: {len(entrenamiento)} vehículos de la parte de entrenamiento")
    return entrenamiento[variables_de_entrada()]


@task(name="lote", cache_policy=NO_CACHE)
def lote(config: dict) -> tuple[pd.DataFrame, dict]:
    datos, meta = leer_lote(ruta_absoluta(config["monitoreo"]["lote_actual"]))
    get_run_logger().info(f"Lote: {meta['nombre']} ({len(datos)} filas, nivel de drift emulado "
                          f"{meta.get('nivel_drift', 'desconocido')})")
    return datos, meta


@task(name="reporte-evidently", cache_policy=NO_CACHE)
def reporte_evidently(datos_referencia: pd.DataFrame, datos_lote: pd.DataFrame, meta: dict, config: dict) -> dict:
    """Ejecuta el preset de drift de Evidently, guarda el reporte HTML y devuelve el resumen por variable."""
    logger = get_run_logger()
    variables = list(datos_referencia.columns)
    snapshot = Report([DataDriftPreset()]).run(reference_data=datos_referencia, current_data=datos_lote[variables])
    carpeta = ruta_absoluta(config["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html = carpeta / f"drift_features_{meta['nombre']}.html"
    snapshot.save_html(str(ruta_html))
    resumen = resumen_evidently(snapshot.dict())
    for c in resumen["columnas"]:
        logger.info(f"{c['variable']:<14} {c['metodo']:<32} {c['valor']:.3f} (umbral {c['umbral']}) "
                    f"-> {'DRIFT' if c['drift'] else 'ok'}")
    logger.info(f"Reporte guardado en {ruta_html}")
    return {**resumen, "reporte": str(ruta_html)}


@task(name="alerta", cache_policy=NO_CACHE)
def alerta(resumen: dict, umbral: float) -> bool:
    """Emite una alerta si la fracción de variables con drift supera el umbral."""
    logger = get_run_logger()
    fraccion = resumen["fraccion"]
    con_drift = [c["variable"] for c in resumen["columnas"] if c["drift"]]
    if hay_alerta(fraccion, umbral):
        logger.warning(f"ALERTA: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%}). "
                       f"Variables con drift: {', '.join(con_drift)}. Conviene revisar el modelo.")
        return True
    logger.info(f"Sin alerta: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%}).")
    return False


@task(name="registro", cache_policy=NO_CACHE)
def registro(resumen: dict, alerta_emitida: bool, meta: dict, config: dict) -> None:
    """Deja el resumen como artefacto de Prefect y, si hay servidor de MLflow, como run de monitoreo."""
    filas = "\n".join(f"| {c['variable']} | {c['metodo']} | {c['valor']:.3f} | {c['umbral']} | "
                      f"{'Sí' if c['drift'] else 'No'} |" for c in resumen["columnas"])
    create_markdown_artifact(
        key="drift-features",
        description=f"Drift de features del lote {meta['nombre']}",
        markdown=(f"# Drift de features: {meta['nombre']}\n\n"
                  f"Nivel de drift emulado: **{meta.get('nivel_drift')}** | Fracción con drift: "
                  f"**{resumen['fraccion']:.0%}** | Umbral: {config['monitoreo']['umbral_drift']:.0%} | "
                  f"Alerta: **{'SÍ' if alerta_emitida else 'no'}**\n\n"
                  "| Variable | Método | Distancia | Umbral | Drift |\n|---|---|---|---|---|\n" + filas),
    )
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment(config["mlflow"]["experimento"])
        with mlflow.start_run(run_name=f"drift-features-{meta['nombre']}"):
            mlflow.set_tags({"fase": "monitoreo", "tipo_drift": "features", "lote": meta["nombre"],
                             "nivel_drift": str(meta.get("nivel_drift")), "alerta": str(alerta_emitida).lower()})
            mlflow.log_param("umbral_drift", config["monitoreo"]["umbral_drift"])
            mlflow.log_metric("fraccion_drift", resumen["fraccion"])
            mlflow.log_metrics({f"distancia_{c['variable']}": c["valor"] for c in resumen["columnas"]})
            mlflow.log_artifact(resumen["reporte"], artifact_path="reporte")


@flow(name="drift-features", log_prints=True)
def drift_features_flow() -> bool:
    config = cargar_config()
    datos_referencia = referencia(config)
    datos_lote, meta = lote(config)
    resumen = reporte_evidently(datos_referencia, datos_lote, meta, config)
    alerta_emitida = alerta(resumen, config["monitoreo"]["umbral_drift"])
    registro(resumen, alerta_emitida, meta, config)
    return alerta_emitida


if __name__ == "__main__":
    drift_features_flow()
