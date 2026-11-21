"""Despliegues en Prefect.

- entrenamiento-precio-vehiculos/entrenamiento: sin horario, se lanza a mano.
- inferencia-precio-vehiculos/inferencia-diaria: todos los días a las 06:00 (hora de Bogotá).
- drift-features/drift-features-diario y drift-evento/drift-evento-diario: todos los días a las 06:30,
  media hora después de la inferencia, sobre el lote que esta acaba de puntuar.

Los horarios están en configs/config.yaml (sección `despliegue`).

Ejecución (desde la raíz del repositorio):
    poetry run python flows/desplegar.py

Deja esta terminal abierta: mientras esté abierta, Prefect puede lanzar las ejecuciones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/ y flows/

from prefect import serve  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from flows.drift_evento_flow import drift_evento_flow  # noqa: E402
from flows.drift_features_flow import drift_features_flow  # noqa: E402
from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.config import cargar_config  # noqa: E402

if __name__ == "__main__":
    despliegue = cargar_config()["despliegue"]
    zona = despliegue["zona_horaria"]
    serve(
        entrenamiento_flow.to_deployment(name="entrenamiento"),
        inferencia_flow.to_deployment(name="inferencia-diaria",
                                      schedule=Cron(despliegue["cron_inferencia"], timezone=zona)),
        drift_features_flow.to_deployment(name="drift-features-diario",
                                          schedule=Cron(despliegue["cron_drift"], timezone=zona)),
        drift_evento_flow.to_deployment(name="drift-evento-diario",
                                        schedule=Cron(despliegue["cron_drift"], timezone=zona)),
    )
