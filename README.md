# Proyecto de Equipo – MLOps

Repositorio del equipo para la asignatura de MLOps (Maestría en Ciencia de Datos, Universidad Santo Tomás).

**Caso de uso:** predecir el **precio final de venta** (`precio_venta`, USD) de vehículos usados multimarca a partir de su marca, tipo de carrocería, año del modelo, kilometraje y precio de lista. Es un problema de **regresión supervisada**.

Los datos son **sintéticos y reproducibles**: se generan con una semilla fija e incluyen relaciones realistas, como la depreciación por antigüedad y kilometraje y los factores de marca y de tipo de vehículo.

**Integrantes:**
- Jhosman Ricardo Moreno Meauri
- Brayan Steven Vera Moreno


---

## Estructura del repositorio

```text
PROYECTO_DE_EQUIPO/
├── configs/
│   └── config.yaml            # Configuración central: datos, modelo y experimento de MLflow
├── data/
│   └── car_sales_dataset.csv  # Dataset sintético (20.000 filas, semilla 42)
├── flows/                     # Flows de Prefect
│   ├── prueba_flow.py         # Validación de los servicios de Prefect y MLflow
│   └── entrenamiento_flow.py  # Pipeline de entrenamiento del equipo
├── model/                     # Reservada para modelos exportados en actividades futuras
├── services/
│   ├── docker-compose.yml     # MLflow (puerto 5000) y Prefect (puerto 4200)
│   └── README.md              # Cómo levantar y detener los servicios
├── src/                       # Código fuente reutilizable
│   ├── config.py              # Lectura de configs/config.yaml
│   ├── generate_data.py       # Generación reproducible del dataset sintético
│   ├── load_data.py           # Carga y validación del dataset
│   └── train.py               # Preprocesado, entrenamiento y métricas
├── tests/                     # Pruebas con pytest (datos y entrenamiento)
├── .gitattributes             # Saltos de línea LF en los CSV (MD5 igual en todos los sistemas)
├── .gitignore
├── poetry.lock                # Versiones exactas instaladas (no se edita a mano)
├── pyproject.toml             # Dependencias declaradas y configuración del proyecto
└── README.md
```

Cada carpeta de código incluye un `__init__.py`. Así Git versiona la carpeta aunque esté vacía, y Python puede importarla como paquete.

---

## Requisitos

| Herramienta | Versión | Comprobación |
|---|---|---|
| Python | 3.12 (el proyecto admite de 3.12 a 3.14) | `python --version` |
| Poetry | 2.x | `poetry --version` |
| Git | reciente | `git --version` |
| Docker Desktop | reciente | `docker --version` |

---

## 1. Clonar e instalar el entorno

```powershell
git clone https://github.com/Bryanv432/PROYECTO_DE_EQUIPO.git
cd PROYECTO_DE_EQUIPO
poetry env use 3.12
poetry install
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
```

Si todo está bien, el último comando imprime `ok`.

- `pyproject.toml` **declara** las dependencias, con versión fijada mediante `==`.
- `poetry.lock` **registra** las versiones exactas de todas las dependencias, incluidas las transitivas. Por eso `poetry install` reproduce el mismo entorno en cualquier computador.
- `package-mode = false` indica que el proyecto no es una librería, así que Poetry solo instala dependencias.
- Solo quien cambie las dependencias ejecuta `poetry add` o `poetry lock`, y luego sube ambos archivos.

## 2. Ejecutar las pruebas

```powershell
poetry run pytest
```

Las pruebas comprueban que:
- La misma semilla genera exactamente el mismo dataset.
- El dataset tiene las columnas y los rangos esperados.
- La carga de datos detecta columnas faltantes.
- El entrenamiento devuelve métricas y es reproducible.

## 3. Generar el dataset (opcional)

El CSV ya está versionado en `data/`. Para regenerarlo:

```powershell
poetry run python src/generate_data.py
```

El script lee `num_registros`, `semilla` y `ruta` de `configs/config.yaml`. Con la semilla 42, el archivo resultante es idéntico al versionado (MD5 `c0b5c77206d5fc3a6537ba3f8150bfcb`). Si el CSV no existe, el pipeline de entrenamiento lo genera automáticamente.

**Estructura del dataset:**

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha_venta` | fecha | Fecha de la venta (2023 a 2025) |
| `marca` | categórica | Toyota, Chevrolet, Renault, Mazda, Nissan, Kia, Ford |
| `tipo_vehiculo` | categórica | Sedán, SUV, Hatchback, Pick-up |
| `año_modelo` | entero | Año de fabricación (2015 a 2025) |
| `kilometraje` | entero | Kilómetros acumulados |
| `precio_lista` | decimal | Precio sugerido de venta (USD) |
| `precio_venta` | decimal | **Variable objetivo**: precio final de cierre (USD) |

## 4. Levantar MLflow y Prefect

Con Docker Desktop abierto:

```powershell
cd services
docker compose up -d
cd ..
```

- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

Más detalles en [`services/README.md`](services/README.md).

## 5. Validar los servicios

En la misma terminal, desde la raíz del repositorio:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/prueba_flow.py
```

Resultado esperado:
- **Prefect:** el run del flow `prueba` aparece en estado **Completed**, con la línea `Suma de 1 a 5 = 15` en los logs.
- **MLflow:** el experimento `prueba-validacion` contiene el run `prueba-mlflow`, con el parámetro `n` y la métrica `suma`.

## 6. Ejecutar el pipeline de entrenamiento

Con las mismas variables de entorno definidas:

```powershell
poetry run python flows/entrenamiento_flow.py
```

El flow `entrenamiento-precio-vehiculos` ejecuta tres tareas:

1. **`cargar-datos`:** lee el CSV, valida sus columnas y registra en los logs de Prefect el tamaño del dataset, su huella MD5 y su rango de fechas.
2. **`entrenar-y-registrar`:** divide los datos en 80 % para entrenamiento y 20 % para prueba, y entrena un Random Forest. En MLflow (experimento `precio-vehiculos-a1`) registra:
   - **Parámetros:** tamaño del dataset, semilla, hiperparámetros y variables.
   - **Métricas:** MAE, RMSE, R² y MAPE.
   - **Trazabilidad:** el dataset de entrada, el tag `dataset_md5` y `config.yaml` como artefacto.
   - **Modelo:** el modelo entrenado, con su firma y un ejemplo de entrada.
3. **`resumen`:** crea en Prefect el artefacto Markdown `resumen-entrenamiento`, con las métricas y el ID del run de MLflow.

Resultado de referencia, con la configuración actual y la semilla 42:

| MAE (USD) | RMSE (USD) | R² | MAPE |
|---|---|---|---|
| 404.91 | 515.89 | 0.9938 | 2.65 % |

**Para probar otro experimento**, cambia `modelo.tipo` o los hiperparámetros, y también `mlflow.nombre_run`, en `configs/config.yaml`. Después vuelve a ejecutar el flow y compara los runs en MLflow.

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `poetry` no se reconoce | Añade al `PATH` la carpeta de scripts que indica el instalador, y luego cierra y vuelve a abrir la terminal. |
| `poetry env use 3.12` falla | Instala Python 3.12 o indica la ruta completa a su `python.exe`. |
| Los flows no aparecen en Prefect o MLflow | Las variables `PREFECT_API_URL` y `MLFLOW_TRACKING_URI` deben estar definidas **en la misma terminal** donde ejecutas el flow. |
| Un puerto está ocupado (5000 o 4200) | Detén el proceso que lo usa, o ejecuta `docker compose down` y luego `docker compose up -d`. |
| `prefect server start` falla con `min_items` | Ocurre si se inicia el servidor desde el entorno de Poetry, por la versión de FastAPI. Levanta el servidor con Docker (paso 4). |

## Tecnologías

Python 3.12 · Poetry 2 · Prefect 3.8.8 · MLflow 3.17.0 · scikit-learn 1.9.1 · pandas 3.0.6 · NumPy 2.5.4 · PyYAML · pytest · Docker Compose · Git/GitHub
