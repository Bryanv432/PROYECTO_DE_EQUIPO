"""API de inferencia: sirve el modelo champion del Model Registry de MLflow.

Ejecución (desde la raíz del repositorio, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn api.main:app --port 8000

- Documentación interactiva: http://localhost:8000/docs
- Estado y versión del modelo: GET http://localhost:8000/health
- Predicción: POST http://localhost:8000/predict

El modelo se carga al arrancar. Si el entrenamiento promueve un champion nuevo,
reinicia la API para que lo use (no hace falta cambiar código).
"""
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.config import cargar_config

CONFIG = cargar_config()
NOMBRE_MODELO = CONFIG["registro"]["nombre_modelo"]
estado: dict = {}


def cargar_champion() -> dict:
    """Carga el modelo con alias champion, su versión y las columnas que espera."""
    uri = f"models:/{NOMBRE_MODELO}@champion"
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion")
    return {
        "modelo": mlflow.sklearn.load_model(uri),
        "version": version.version,
        "features": version.tags.get("features", "desconocido"),
        "columnas": mlflow.models.get_model_info(uri).signature.inputs.input_names(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado.update(cargar_champion())
    yield
    estado.clear()


app = FastAPI(
    title="API de precio de vehículos usados",
    description="Estima el precio de venta (USD) de un vehículo usado con el modelo champion de MLflow.",
    version="2.0.0",
    lifespan=lifespan,
)


class Vehiculo(BaseModel):
    """Datos de un vehículo. Se envían todos; el modelo usa solo las variables con las que se entrenó."""
    marca: str = Field(examples=["Toyota"], description="Toyota, Chevrolet, Renault, Mazda, Nissan, Kia o Ford")
    tipo_vehiculo: str = Field(examples=["SUV"], description="Sedán, SUV, Hatchback o Pick-up")
    año_modelo: int = Field(examples=[2020], ge=1990, le=2030)
    kilometraje: int = Field(examples=[60000], ge=0)
    precio_lista: float = Field(examples=[22000.0], gt=0, description="Precio publicado en USD")


class Prediccion(BaseModel):
    precio_predicho: float = Field(description="Precio de venta estimado en USD")
    version_modelo: str
    features: str = Field(description="Conjunto de variables del champion (ALL, NUMERIC o CUSTOM)")


@app.get("/health")
def health() -> dict:
    """Estado del servicio y versión del modelo que está sirviendo."""
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": estado["version"], "features": estado["features"]}


@app.post("/predict", response_model=Prediccion)
def predict(vehiculo: Vehiculo) -> Prediccion:
    """Estima el precio de venta de un vehículo."""
    datos = pd.DataFrame([vehiculo.model_dump()])[estado["columnas"]]
    precio = float(estado["modelo"].predict(datos)[0])
    return Prediccion(precio_predicho=round(precio, 2), version_modelo=estado["version"], features=estado["features"])
