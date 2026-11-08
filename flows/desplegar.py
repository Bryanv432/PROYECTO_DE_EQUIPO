"""Despliegues en Prefect de las dos pipelines.

- entrenamiento-precio-vehiculos/entrenamiento: sin horario, se lanza a mano.
- inferencia-precio-vehiculos/inferencia-diaria: todos los días a la hora de configs/config.yaml
  (por defecto 06:00, hora de Bogotá).

Ejecución (desde la raíz del repositorio):
    poetry run python flows/desplegar.py

Deja esta terminal abierta: mientras esté abierta, Prefect puede lanzar las ejecuciones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar src/ y flows/

from prefect import serve  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.config import cargar_config  # noqa: E402

if __name__ == "__main__":
    despliegue = cargar_config()["despliegue"]
    serve(
        entrenamiento_flow.to_deployment(name="entrenamiento"),
        inferencia_flow.to_deployment(
            name="inferencia-diaria",
            schedule=Cron(despliegue["cron_inferencia"], timezone=despliegue["zona_horaria"]),
        ),
    )
