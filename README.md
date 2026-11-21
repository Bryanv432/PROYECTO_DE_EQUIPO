# Proyecto de Equipo – MLOps

Repositorio del equipo para la asignatura de MLOps (Maestría en Ciencia de Datos, Universidad Santo Tomás).

**Caso de uso:** estimar el **precio final de venta** (`precio_venta`, USD) de vehículos usados multimarca a partir de su marca, tipo de carrocería, año del modelo, kilometraje y precio de lista. Es un problema de **regresión supervisada** con datos sintéticos reproducibles (semilla 42).

**Integrantes:**
- Jhosman Ricardo Moreno Meauri
- Brayan Steven Vera Moreno

---

## Estructura del repositorio

```text
PROYECTO_DE_EQUIPO/
├── .github/workflows/ci.yml   # CI: ejecuta los tests en cada pull request (actividad 3)
├── api/
│   └── main.py                # API de FastAPI que sirve el modelo champion (actividad 2)
├── configs/
│   ├── config.yaml            # datos, modelo, champion, inferencia (nivel de drift), monitoreo y despliegue
│   └── features.yaml          # selección de variables del experimento: ALL, NUMERIC o CUSTOM
├── data/
│   ├── car_sales_dataset.csv  # dataset sintético (20.000 filas, semilla 42)
│   ├── lotes/                 # copia de cada lote puntuado, con drift emulado (no se versiona)
│   └── predicciones/          # prediccion_<fecha>.csv de la inferencia (no se versiona)
├── flows/                     # flows de Prefect
│   ├── prueba_flow.py         # validación de servicios (actividad 1)
│   ├── entrenamiento_flow.py  # ingesta → entrenamiento → evaluación → registro
│   ├── inferencia_flow.py     # lote de prueba + drift emulado → predicción → copia del lote
│   ├── drift_features_flow.py # drift de las variables de entrada con Evidently (actividad 3)
│   ├── drift_evento_flow.py   # drift del precio de venta con la prueba KS (actividad 3)
│   └── desplegar.py           # despliegues en Prefect
├── reports/                   # reportes de drift por lote (no se versionan)
├── services/                  # MLflow y Prefect con Docker Compose
├── src/                       # lógica del proyecto, sin Prefect ni MLflow
│   ├── config.py              # lectura de los YAML
│   ├── generate_data.py       # generación reproducible del dataset
│   ├── load_data.py           # carga y validación del dataset
│   ├── features.py            # resuelve ALL / NUMERIC / CUSTOM
│   ├── train.py               # pipeline de scikit-learn y métricas
│   ├── champion.py            # regla para decidir si un modelo pasa a champion
│   └── drift.py               # emulación y detección de drift (actividad 3)
├── tests/                     # pruebas con pytest (datos, modelo, inferencia, API y drift)
├── pytest.ini
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## Requisitos

| Herramienta | Versión |
|---|---|
| Python | 3.12, 3.13 o 3.14 |
| Poetry | 2.x |
| Git | reciente |
| Docker Desktop | reciente |

---

## 1. Instalar el entorno

```powershell
git clone https://github.com/Bryanv432/PROYECTO_DE_EQUIPO.git
cd PROYECTO_DE_EQUIPO
py -0   # muestra las versiones de Python instaladas
poetry env use (py -3.14 -c "import sys; print(sys.executable)")   # usa tu versión: 3.12, 3.13 o 3.14
poetry install
poetry run pytest
```

`poetry run pytest` debe terminar con todas las pruebas en verde.

## 2. Levantar MLflow y Prefect

```powershell
cd services
docker compose up -d
cd ..
```

- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

En **cada terminal** donde ejecutes un flow o la API, desde la raíz del repositorio:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

`PYTHONUTF8` evita un error de codificación de la consola de Windows con los mensajes de MLflow.

---

## Actividad 1: validación de servicios

```powershell
poetry run python flows/prueba_flow.py
```

El run `prueba` queda en estado Completed en Prefect, y el experimento `prueba-validacion` aparece en MLflow.

---

## Actividad 2: entrenamiento, inferencia, despliegue y API

### Configuración

Todo lo que puede cambiar está en YAML, no en el código:

| Archivo | Sección | Para qué |
|---|---|---|
| `configs/config.yaml` | `modelo` | Tipo de modelo (`random_forest`, `gradient_boosting`, `linear_regression`) y sus hiperparámetros |
| `configs/config.yaml` | `registro` | Nombre del modelo en MLflow y reglas del champion |
| `configs/config.yaml` | `inferencia` | Tamaño del lote y carpeta de salida |
| `configs/config.yaml` | `despliegue` | Horario de la inferencia diaria y zona horaria |
| `configs/features.yaml` | `seleccion` | Variables del experimento: `ALL`, `NUMERIC` o `CUSTOM` (las de `custom`) |

**Por qué R² y no AUC.** La guía de la actividad compara modelos con AUC, que es una métrica de clasificación. Nuestro problema es de regresión, así que usamos **R²**: igual que el AUC, va de 0 a 1 y es mejor cuanto más alto.

**Reglas del champion** (sección `registro`). Un modelo nuevo se registra como versión nueva y recibe el alias `champion` solo si cumple las tres condiciones:
1. Su R² alcanza el mínimo (`umbral_minimo: 0.90`).
2. Su MAE no pasa de `mae_maximo: 1000` USD. Es un criterio adicional del grupo.
3. Supera al champion vigente por al menos `mejora_minima: 0.001`. También es un criterio del grupo, para no cambiar de modelo por diferencias de ruido.

Si no cumple alguna, el run queda en MLflow con la etiqueta `promovido = false` y el motivo, pero el alias no cambia.

### 1. Entrenamiento

```powershell
poetry run python flows/entrenamiento_flow.py
```

El flow `entrenamiento-precio-vehiculos` tiene cuatro tareas: `ingesta` → `entrenamiento` → `evaluacion` → `registro`.

En MLflow, el experimento `precio-vehiculos` recibe un run llamado `entrenamiento-<modelo>-<features>`. El run guarda:
- **Etiquetas:** `fase = entrenamiento`, `dataset_md5`, `promovido` y `motivo`.
- **Parámetros:** incluidos `features` (el alias) y `n_features`.
- **Métricas:** `r2`, `mae`, `rmse` y `mape`.
- **Artefactos:** los dos YAML y el modelo.

Si el modelo pasa a champion, aparece en **Models → precio-vehiculos-modelo** con el alias `@champion` y la etiqueta `estado = aprobado`.

### 2. Inferencia

```powershell
poetry run python flows/inferencia_flow.py
```

El flow `inferencia-precio-vehiculos` hace cuatro cosas:
1. Carga `models:/precio-vehiculos-modelo@champion`, por alias y nunca por número de versión.
2. Toma un lote de 500 vehículos de la **parte de prueba** y le aplica el drift emulado (ver la actividad 3).
3. Predice su precio y guarda `data/predicciones/prediccion_<fecha>.csv`, con el precio predicho y el error frente al precio real.
4. Registra un run `inferencia-<fecha>-drift-<nivel>` con `fase = inferencia`, `modelo_version`, `nivel_drift`, `filas_predichas`, `precio_medio_predicho` y `mae_lote`.

### 3. Despliegues en Prefect

```powershell
poetry run python flows/desplegar.py
```

Deja esa terminal abierta. Quedan cuatro despliegues separados:
- `entrenamiento-precio-vehiculos/entrenamiento`: sin horario, se lanza a mano.
- `inferencia-precio-vehiculos/inferencia-diaria`: todos los días a las **06:00, hora de Bogotá** (cron `0 6 * * *`).
- `drift-features/drift-features-diario` y `drift-evento/drift-evento-diario`: todos los días a las **06:30**, sobre el lote que acaba de puntuar la inferencia (actividad 3).

### 4. API

En otra terminal, con las variables definidas:

```powershell
poetry run uvicorn api.main:app --port 8000
```

- Documentación interactiva: http://localhost:8000/docs
- Estado y versión del modelo: http://localhost:8000/health

#### Predicción de ejemplo

**Petición** `POST http://localhost:8000/predict`:

```json
{
  "marca": "Toyota",
  "tipo_vehiculo": "SUV",
  "año_modelo": 2020,
  "kilometraje": 60000,
  "precio_lista": 22000.0
}
```

Desde PowerShell, en una sola línea:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json; charset=utf-8" -Body '{"marca":"Toyota","tipo_vehiculo":"SUV","año_modelo":2020,"kilometraje":60000,"precio_lista":22000.0}'
```

**Respuesta** (con la versión 2 del modelo, entrenada con `NUMERIC`):

```json
{
  "precio_predicho": 21368.57,
  "version_modelo": "2",
  "features": "NUMERIC"
}
```

La API recibe todos los datos del vehículo y usa solo las variables con las que se entrenó el champion. Si el entrenamiento promueve un champion nuevo, **reinicia la API** (Ctrl + C y vuelve a lanzarla) para que lo use. No hay que cambiar código.

### Demostración del criterio de éxito

Con la semilla 42, esta secuencia da siempre el mismo resultado:

| Paso | `seleccion` | R² | Resultado |
|---|---|---|---|
| 1 | `CUSTOM` | 0,9871 | Versión 1 → champion |
| 2 | `NUMERIC` | 0,9937 | Versión 2 → champion (mejora a la 1) |
| 3 | `ALL` | 0,9938 | No se promueve: mejora en menos de 0,001 |

Después del paso 2, al reiniciar la API, `/health` pasa de `"version": "1"` a `"version": "2"`.

---

## Actividad 3: monitoreo, drift y CI

### Drift emulado en la inferencia

La inferencia toma cada lote de la **parte de prueba** del dataset, es decir, de vehículos que el modelo no vio al entrenar. Es la misma partición del entrenamiento: 20 % con semilla 42. Antes de predecir, al lote se le aplica un drift emulado. El nivel lo fija `inferencia.nivel_drift` en `configs/config.yaml`; no lo introduce quien ejecuta la pipeline.

El nivel decide **cuántas** de las 5 variables de entrada cambian (`int(nivel × 5)`), siempre en este orden:

| Orden | Variable | Cambio emulado | Qué representa |
|---|---|---|---|
| 1 | `kilometraje` | × 1,6 | Llegan vehículos más usados |
| 2 | `año_modelo` | − 4 años | Llegan modelos más antiguos |
| 3 | `tipo_vehiculo` | 50 % pasa a Pick-up | Cambia la mezcla de carrocerías |
| 4 | `marca` | 50 % pasa a Ford | Cambia la mezcla de marcas |
| 5 | `precio_lista` | × 1,3 | Suben los precios publicados |

Además, la **variable objetivo** `precio_venta` se multiplica por `1 + nivel × 0,25`, como una inflación de los precios de cierre.

| `nivel_drift` | Variables que cambian | Fracción | Precio de venta |
|---|---|---|---|
| 0.0 | ninguna | 0 % | sin cambio |
| 0.3 | kilometraje | 20 % | +7,5 % |
| 0.6 | kilometraje, año_modelo, tipo_vehiculo | 60 % | +15 % |

Cada ejecución guarda el lote puntuado (con el precio real) en dos archivos. El primero es `data/lotes/lote_actual.csv`, que leen los pipelines de drift. El segundo es una copia con fecha y nivel, `data/lotes/lote_<fecha>_nivel_<nivel>.csv`.

### Pipelines de detección de drift

Se ejecutan después de la inferencia y comparan el lote con la parte de entrenamiento. La lógica no depende de MLflow; si `MLFLOW_TRACKING_URI` está definida, además registran un run con `fase = monitoreo`.

```powershell
poetry run python flows/drift_features_flow.py
poetry run python flows/drift_evento_flow.py
```

| Pipeline | Qué compara | Cómo | Alerta | Reporte |
|---|---|---|---|---|
| `drift-features` | Las 5 variables de entrada | Evidently `DataDriftPreset`: Wasserstein (numéricas) y Jensen-Shannon (categóricas), umbral 0,1 por variable | Si la fracción de variables con drift supera `monitoreo.umbral_drift` (0,3) | `reports/drift_features_<lote>.html` |
| `drift-evento` | El precio de venta (target) | Prueba de Kolmogorov-Smirnov de dos muestras | Si el p-valor es menor que `monitoreo.alfa_evento` (0,05) | `reports/drift_evento_<lote>.md` |

La alerta es un log de nivel **WARNING** que empieza con `ALERTA:` y se ve en la consola y en los logs de Prefect. Los dos flows también dejan un artefacto de resumen en Prefect.

**Por qué KS y no chi-cuadrado:** el ejemplo de referencia compara una tasa binaria (churn sí o no). Nuestro objetivo es un precio continuo, así que comparamos su distribución completa con la prueba de Kolmogorov-Smirnov.

**Lotes sin etiquetas:** en producción el precio real llega con retraso. Si el lote no trae `precio_venta`, o le faltan valores, `drift-evento` se detiene con un mensaje claro y no calcula nada.

Resultados con la semilla 42 (lote de 500 vehículos):

| Nivel | Drift de features | Alerta features | Target (media, p-valor KS) | Alerta target | Error medio del lote |
|---|---|---|---|---|---|
| 0.0 | 0 % | No | −3,1 %, p = 0,097 | No | 412 USD |
| 0.3 | 20 % | No | +4,2 %, p = 0,048 | Sí | 1.126 USD |
| 0.6 | 60 % | **Sí** | +11,4 %, p < 0,0001 | Sí | 2.253 USD |

### Tests y CI

```powershell
poetry run pytest
```

Son 62 tests que no necesitan MLflow ni Prefect levantados:

| Archivo | Qué prueba |
|---|---|
| `tests/test_datos.py` | Esquema, tipos, rangos, categorías, nulos, duplicados y reproducibilidad del dataset |
| `tests/test_features.py`, `tests/test_train.py`, `tests/test_champion.py` | Selección de variables, entrenamiento, métricas y regla del champion |
| `tests/test_inferencia.py` | Que el lote salga solo de la parte de prueba, el nivel de drift, la copia del lote y la predicción (con un modelo falso) |
| `tests/test_api.py` | `/health` y `/predict` con `TestClient` y un modelo falso (sin MLflow) |
| `tests/test_drift.py` | Emulación, rechazo de lotes sin etiquetas, prueba KS, lectura de Evidently, alerta y el criterio de éxito (0.0 sin alerta, 0.6 con alerta) |

El workflow `.github/workflows/ci.yml` ejecuta los mismos tests en GitHub Actions en cada pull request y en cada push a `main`.

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `RESOURCE_DOES_NOT_EXIST` o «Registered Model ... not found» en la inferencia o la API | Todavía no hay champion. Ejecuta el entrenamiento primero. |
| Los runs no aparecen en Prefect o MLflow | Define las variables `$env:` en la misma terminal donde ejecutas el flow. |
| `UnicodeEncodeError: 'charmap' codec` | Falta `$env:PYTHONUTF8 = "1"` en esa terminal. |
| La API sigue respondiendo con la versión anterior | Reiníciala: carga el champion solo al arrancar. |
| Los despliegues no se ejecutan | La terminal de `desplegar.py` debe seguir abierta. |
| `Could not find the python executable 3.12` | Usa `poetry env use (py -3.14 -c "import sys; print(sys.executable)")`, cambiando 3.14 por tu versión. |
| `No existe el lote data/lotes/lote_actual.csv` | Los pipelines de drift leen el lote de la inferencia: ejecuta primero `flows/inferencia_flow.py`. |
| `El lote no tiene la columna 'precio_venta'` | Es el comportamiento esperado con lotes sin etiquetas: el drift del target necesita el precio real. |
| Python 3.15 instalado | El proyecto admite de 3.12 a 3.14. Instala también la 3.14 (`winget install Python.Python.3.14`) y usa `py -3.14`. |
