# Servicios: MLflow y Prefect

Los dos servicios corren con Docker Compose, a partir de imágenes oficiales con la versión fijada:

| Servicio | Imagen | URL |
|---|---|---|
| MLflow (tracking y registro de modelos) | `ghcr.io/mlflow/mlflow:v3.17.0` | http://localhost:5000 |
| Prefect (orquestación de flows) | `prefecthq/prefect:3.8.8-python3.12` | http://localhost:4200 |

## Levantar y detener

Desde esta carpeta (`services/`), con Docker Desktop abierto:

```powershell
docker compose up -d      # levanta los servicios en segundo plano
docker compose ps         # comprueba que los dos contenedores están "running"
docker compose down       # detiene los servicios; los datos se conservan
```

## Dónde se guardan los datos

- **MLflow:** la base SQLite y los artefactos quedan en `services/mlflow_store/`. Esa carpeta está en `.gitignore`, así que no se versiona.
- **Prefect:** su base de datos queda en el volumen de Docker `prefect_data`.

## Conectar los flows a los servicios

En la terminal donde se ejecutan los flows, desde la raíz del repositorio:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```
