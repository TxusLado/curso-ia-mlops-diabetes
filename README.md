# Proyecto final — MLOps con el dataset Diabetes

**Curso IA 2026 · Proyecto educativo de Machine Learning en producción**

Este proyecto recorre el ciclo de vida de un modelo de regresión: preparar y validar datos, entrenar candidatos, comparar resultados, registrar el mejor modelo en MLflow, evaluarlo mediante un *Quality Gate*, servirlo con FastAPI y comprobar automáticamente todo el flujo con GitHub Actions.

> **Aviso:** se utiliza el dataset público `load_diabetes` de scikit-learn como ejercicio. **No es un sistema de diagnóstico ni un servicio apto para decisiones clínicas.**

## 1. Objetivo

Predecir la variable `target` del dataset de diabetes con un modelo de regresión y, sobre todo, demostrar un proceso **reproducible, verificable y trazable**.

Se han utilizado Python, Pandas, Pydantic, scikit-learn, MLflow, SQLite, FastAPI, Pytest y GitHub Actions.

## 2. Flujo de trabajo

1. **Extracción:** obtener los 442 registros de `load_diabetes` y guardarlos en `data/raw/`.
2. **Validación:** comprobar las 11 columnas (10 características y `target`) con Pydantic, fila a fila.
3. **Limpieza:** eliminar valores nulos o registros duplicados si aparecen y guardar el resultado en `data/processed/`.
4. **Ingeniería de características y auditoría:** crear dos interacciones (`bmi_age_interaction` y `bp_bmi_interaction`) y revisar valores extremos. Se conserva el conjunto de datos completo para el ejercicio.
5. **Entrenamiento:** comparar `LinearRegression`, `RandomForestRegressor` y `GradientBoostingRegressor` mediante validación cruzada de cinco particiones.
6. **Registro MLflow:** guardar los experimentos y registrar la mejor ejecución como `DiabetesRegressionModel`.
7. **Quality Gate:** comparar el RMSE del modelo registrado con un `DummyRegressor` (media). Solo se asigna el alias `champion` si supera esta referencia.
8. **Servicio FastAPI:** cargar el `champion` una sola vez al iniciar y exponer predicciones y metadatos.
9. **Pruebas y CI:** reconstruir el proceso y ejecutar Pytest automáticamente en GitHub Actions.
10. **Trazabilidad básica:** identificar cada petición y registrar fecha, resultado HTTP, duración y, en las predicciones, versión y ejecución MLflow.

## 3. Resultados del experimento

En la ejecución inicial del proyecto, las medias aproximadas de RMSE en validación cruzada fueron:

| Candidato | RMSE CV (menor es mejor) |
|---|---:|
| Regresión lineal | **55,41** |
| Random Forest | 59,14 |
| Gradient Boosting | 60,55 |

Se seleccionó **regresión lineal**. En el conjunto reservado de prueba (89 registros) se obtuvo aproximadamente:

- **RMSE:** 52,49.
- **MAE:** 41,71.
- **R²:** 0,48.
- **RMSE del baseline:** 73,22.

El modelo superó la puerta de calidad educativa y recibió el alias `champion`. Estos resultados describen una ejecución del ejercicio; pueden variar ligeramente entre entornos.

## 4. Estructura del repositorio

```text
data/
  raw/diabetes.csv
  processed/diabetes_clean.csv
reports/
  outliers.csv
src/
  data/
    extract.py
    validate.py
    clean.py
    features.py
    preprocess.py
  models/
    train.py
    register_model.py
    evaluate_model.py
  api/
    main.py
    model.py
tests/
  test_api.py
.github/workflows/
  ci.yml
requirements.txt
.gitignore
README.md
```

Los archivos `mlflow.db` y `mlruns/` se generan localmente y están excluidos de Git. **No deben confundirse con los archivos de código ni borrarse sin conservar un respaldo de los experimentos.**

## 5. Reproducir el proyecto

Requisitos: **Python 3.11** y Git. Ejecutar los comandos desde la carpeta raíz del repositorio.

```bash
python -m venv .venv
```

Activar el entorno:

- Linux/macOS: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

Instalar dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ejecutar las etapas **en este orden**:

```bash
python -m src.data.extract
python -m src.data.validate
python -m src.data.clean
python -m src.data.preprocess

python -m src.models.train
python -m src.models.register_model
python -m src.models.evaluate_model

python -m pytest tests/test_api.py -v
```

La evaluación necesita el modelo previamente registrado. En un entorno nuevo, el *Quality Gate* debe completarse correctamente para crear el alias `champion` que utiliza FastAPI.

> La automatización de `.github/workflows/ci.yml` ejecuta estos mismos pasos en una máquina Ubuntu limpia cuando hay cambios en `main`, en propuestas de cambios o cuando se inicia manualmente.

## 6. Ejecutar FastAPI

Después de reconstruir y evaluar el modelo:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: **http://127.0.0.1:8000/docs**

| Ruta | Método | Función |
|---|---|---|
| `/health` | GET | Comprobar que la API responde. |
| `/model-info` | GET | Consultar nombre, alias, versión y `run_id`. |
| `/predict` | POST | Predecir a partir de las diez características originales. |

La ruta `/predict` recibe los campos `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5` y `s6`. Dentro del servicio se calculan las dos características de interacción para mantener el mismo contrato de 12 columnas que en el entrenamiento.

Para enviar un registro del CSV de prueba al servicio local:

```python
import pandas as pd
import httpx

cols = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
registro = pd.read_csv("data/processed/diabetes_clean.csv").iloc[0]
entrada = {col: float(registro[col]) for col in cols}

r = httpx.post("http://127.0.0.1:8000/predict", json=entrada)
print("HTTP:", r.status_code)
print("Request ID:", r.headers.get("X-Request-ID"))
print("Respuesta:", r.json())
```

Esta prueba requiere mantener Uvicorn en ejecución en otra terminal.

## 7. Validación, errores y trazabilidad

Las pruebas automáticas comprueban los endpoints y casos como:

- Predicción correcta: **HTTP 200**.
- Datos de tipo incorrecto, ausentes, adicionales o no finitos: **HTTP 422**.
- Fallo simulado del modelo: **HTTP 500** con un mensaje controlado.
- Consulta de metadatos: versión, alias y referencia de ejecución.
- **Request ID** único por petición, incluso cuando la entrada es incorrecta.

Cada respuesta incluye una cabecera `X-Request-ID`. Los logs del servicio registran automáticamente la hora, ese identificador, el estado HTTP y la duración de la petición. Cuando hay una predicción válida, registran también el valor predicho, la versión del modelo y el `run_id` de MLflow. No se escriben las diez características recibidas en los logs.

**Alcance real:** estos logs permiten una investigación básica de incidencias, pero no son todavía una plataforma de observabilidad completa. No hay dashboard, alertas automáticas, detección continua de deriva ni infraestructura de despliegue productivo.

## 8. Integración continua

GitHub Actions ejecuta `CI - Diabetes MLOps` para verificar de principio a fin la preparación de datos, entrenamiento, registro, evaluación y pruebas de la API.

En cada ejecución limpia se regeneran la base de datos SQLite de MLflow y sus artefactos. Si falla una etapa, el flujo no llega a aprobar la ejecución completa.

La pestaña **Actions** del repositorio permite consultar el resultado y los logs de cada etapa.

## 9. Limitaciones y mejoras posibles

- El *Quality Gate* compara el campeón con un modelo de referencia sencillo: **no certifica utilidad clínica**.
- La prueba de API depende de que antes se haya reconstruido y registrado el campeón.
- SQLite y el almacenamiento local de artefactos son adecuados para este proyecto educativo; un servicio distribuido requeriría otro almacenamiento y una arquitectura más robusta.
- No se han añadido autenticación de usuarios, limitación de peticiones, despliegue permanente, dashboard o reentrenamiento automático por deriva.

**Conclusión:** el resultado es un prototipo educativo reproducible que integra varias fases reales del ciclo de vida MLOps, no un servicio clínico ni una plataforma productiva completa.
