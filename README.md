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
├── api/
│   └── main.py                # API de FastAPI que sirve el modelo champion (actividad 2)
├── configs/
│   ├── config.yaml            # datos, modelo, reglas del champion, inferencia y despliegue
│   └── features.yaml          # selección de variables del experimento: ALL, NUMERIC o CUSTOM
├── data/
│   ├── car_sales_dataset.csv  # dataset sintético (20.000 filas, semilla 42)
│   └── predicciones/          # prediccion_<fecha>.csv de la inferencia diaria (no se versiona)
├── flows/                     # flows de Prefect
│   ├── prueba_flow.py         # validación de servicios (actividad 1)
│   ├── entrenamiento_flow.py  # ingesta → entrenamiento → evaluación → registro
│   ├── inferencia_flow.py     # inferencia con el modelo champion
│   └── desplegar.py           # despliegues en Prefect (inferencia diaria)
├── services/                  # MLflow y Prefect con Docker Compose
├── src/                       # lógica del proyecto, sin Prefect ni MLflow
│   ├── config.py              # lectura de los YAML
│   ├── generate_data.py       # generación reproducible del dataset
│   ├── load_data.py           # carga y validación del dataset
│   ├── features.py            # resuelve ALL / NUMERIC / CUSTOM
│   ├── train.py               # pipeline de scikit-learn y métricas
│   └── champion.py            # regla para decidir si un modelo pasa a champion
├── tests/                     # pruebas con pytest
│   ├── test_api.py            # pruebas de endpoints FastAPI (/health, /model-info, /predict)
│   ├── test_champion.py       # pruebas de reglas de decisión de champion
│   ├── test_datos.py          # pruebas de reproducibilidad y datos
│   ├── test_features.py       # pruebas de alias ALL, NUMERIC y CUSTOM
│   └── test_train.py          # pruebas del pipeline de entrenamiento
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
2. Toma 100 vehículos al azar como lote del día.
3. Predice su precio y guarda `data/predicciones/prediccion_<fecha>.csv`, con el precio predicho y el error frente al precio real.
4. Registra un run `inferencia-<fecha>` con `fase = inferencia`, `modelo_version`, `filas_predichas`, `precio_medio_predicho` y `mae_lote`.

### 3. Despliegues en Prefect

```powershell
poetry run python flows/desplegar.py
```

Deja esa terminal abierta. Quedan dos despliegues separados:
- `entrenamiento-precio-vehiculos/entrenamiento`: sin horario, se lanza a mano.
- `inferencia-precio-vehiculos/inferencia-diaria`: todos los días a las **06:00, hora de Bogotá** (cron `0 6 * * *`).

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

#### `configs/features.yaml`:

Archivo de configuración para definir el conjunto de variables del experimento sin alterar el código fuente. En cada ejecución (*run*) se registran automáticamente en MLflow el alias seleccionado (`features`) y la cantidad de variables utilizadas (`n_features`).

```yaml
# Variables del experimento.
# Opciones para `seleccion`: ALL | NUMERIC | CUSTOM
seleccion: CUSTOM

numericas:
  - año_modelo
  - kilometraje
  - precio_lista

categoricas:
  - marca
  - tipo_vehiculo

# Subconjunto personalizado (ej. estimar el precio sin `precio_lista`)
custom:
  - marca
  - tipo_vehiculo
  - año_modelo
  - kilometraje
```
- ALL: utiliza todas las variables disponibles (numéricas y categóricas).

- NUMERIC: excluye las variables categóricas y entrena únicamente con las numéricas.

- CUSTOM: utiliza estrictamente la lista definida bajo la clave custom (nombres exactos del CSV).

### Demostración del criterio de éxito

Con la semilla 42, esta secuencia da siempre el mismo resultado:

| Paso | `seleccion` | R² | Resultado |
|---|---|---|---|
| 1 | `CUSTOM` | 0,9871 | Versión 1 → champion |
| 2 | `NUMERIC` | 0,9937 | Versión 2 → champion (mejora a la 1) |
| 3 | `ALL` | 0,9938 | No se promueve: mejora en menos de 0,001 |

Después del paso 2, al reiniciar la API, `/health` pasa de `"version": "1"` a `"version": "2"`.

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
