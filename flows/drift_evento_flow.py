"""Pipeline de drift de evento (target): compara el precio de venta del lote con el del entrenamiento.

Pasos: referencia -> lote (con etiquetas) -> prueba KS -> alerta -> registro.

- El lote debe traer las etiquetas reales (`precio_venta`). En producción llegan con retraso: si faltan,
  el pipeline se detiene con un mensaje claro en lugar de calcular algo sin sentido.
- Como el objetivo es continuo, se usa la prueba de Kolmogorov-Smirnov de dos muestras.
- Si el p-valor es menor que `monitoreo.alfa_evento`, emite una ALERTA (log de nivel WARNING).
- Guarda un reporte en Markdown por lote en `reports/` y, si hay servidor de MLflow, un run de monitoreo.

Ejecución (desde la raíz del repositorio, después de la inferencia):
    poetry run python flows/drift_evento_flow.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.drift import dividir_entrenamiento_prueba, drift_de_target, leer_lote, validar_lote  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


@task(name="referencia-target", cache_policy=NO_CACHE)
def referencia(config: dict) -> pd.Series:
    entrenamiento, _ = dividir_entrenamiento_prueba(cargar_datos(config["datos"]),
                                                    config["division"]["test_size"], config["division"]["semilla"])
    return entrenamiento[config["datos"]["objetivo"]]


@task(name="lote-con-etiquetas", cache_policy=NO_CACHE)
def lote(config: dict) -> tuple[pd.DataFrame, dict]:
    """Lee el lote y comprueba que trae las etiquetas reales; si no, falla con un mensaje claro."""
    datos, meta = leer_lote(ruta_absoluta(config["monitoreo"]["lote_actual"]))
    validar_lote(datos, config["datos"]["objetivo"])
    get_run_logger().info(f"Lote {meta['nombre']}: {len(datos)} filas con etiquetas reales")
    return datos, meta


@task(name="prueba-ks", cache_policy=NO_CACHE)
def evaluar(referencia_objetivo: pd.Series, datos_lote: pd.DataFrame, config: dict) -> dict:
    resultado = drift_de_target(referencia_objetivo, datos_lote[config["datos"]["objetivo"]],
                                config["monitoreo"]["alfa_evento"])
    get_run_logger().info(
        f"Precio de venta medio: entrenamiento {resultado['media_referencia']:,.0f} USD, lote "
        f"{resultado['media_lote']:,.0f} USD ({resultado['cambio_media']:+.1%}) | KS = "
        f"{resultado['estadistico_ks']:.3f}, p-valor = {resultado['p_valor']:.4g}")
    return resultado


@task(name="alerta", cache_policy=NO_CACHE)
def alerta(resultado: dict, alfa: float) -> bool:
    logger = get_run_logger()
    if resultado["drift"]:
        logger.warning(f"ALERTA: drift del precio de venta (p-valor {resultado['p_valor']:.4g} < {alfa}). "
                       f"La media cambió {resultado['cambio_media']:+.1%} frente al entrenamiento.")
        return True
    logger.info(f"Sin alerta: el precio de venta no muestra drift (p-valor {resultado['p_valor']:.4g} >= {alfa}).")
    return False


@task(name="registro", cache_policy=NO_CACHE)
def registro(resultado: dict, alerta_emitida: bool, meta: dict, config: dict) -> Path:
    """Guarda el reporte del lote en reports/ y lo deja en Prefect y, si hay servidor, en MLflow."""
    texto = (f"# Drift del target: {meta['nombre']}\n\n"
             f"Nivel de drift emulado: **{meta.get('nivel_drift')}** | Alerta: **{'SÍ' if alerta_emitida else 'no'}**\n\n"
             "| Medida | Entrenamiento | Lote |\n|---|---|---|\n"
             f"| Precio de venta medio (USD) | {resultado['media_referencia']:,.0f} | {resultado['media_lote']:,.0f} |\n\n"
             f"- Cambio de la media: {resultado['cambio_media']:+.1%}\n"
             f"- Estadístico KS: {resultado['estadistico_ks']:.3f}\n"
             f"- p-valor: {resultado['p_valor']:.4g} (alfa {config['monitoreo']['alfa_evento']})\n")
    carpeta = ruta_absoluta(config["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"drift_evento_{meta['nombre']}.md"
    ruta.write_text(texto, encoding="utf-8")
    create_markdown_artifact(key="drift-evento", markdown=texto, description=f"Drift del target del lote {meta['nombre']}")
    get_run_logger().info(f"Reporte guardado en {ruta}")
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment(config["mlflow"]["experimento"])
        with mlflow.start_run(run_name=f"drift-evento-{meta['nombre']}"):
            mlflow.set_tags({"fase": "monitoreo", "tipo_drift": "evento", "lote": meta["nombre"],
                             "nivel_drift": str(meta.get("nivel_drift")), "alerta": str(alerta_emitida).lower()})
            mlflow.log_param("alfa_evento", config["monitoreo"]["alfa_evento"])
            mlflow.log_metrics({k: v for k, v in resultado.items() if k != "drift"})
            mlflow.log_artifact(str(ruta), artifact_path="reporte")
    return ruta


@flow(name="drift-evento", log_prints=True)
def drift_evento_flow() -> bool:
    config = cargar_config()
    objetivo_referencia = referencia(config)
    datos_lote, meta = lote(config)
    resultado = evaluar(objetivo_referencia, datos_lote, config)
    alerta_emitida = alerta(resultado, config["monitoreo"]["alfa_evento"])
    registro(resultado, alerta_emitida, meta, config)
    return alerta_emitida


if __name__ == "__main__":
    drift_evento_flow()
