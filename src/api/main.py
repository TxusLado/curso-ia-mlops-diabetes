from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from src.api.model import (
    MODEL_NAME,
    load_champion_model,
    get_model,
    get_model_info,
)
from src.data.features import create_features


# Dejar un registro con fecha/hora tanto de las peticiones como de los errores.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
logger.propagate = False

# Columnas que espera nuestro modelo campeon: 10 originales + 2 interacciones.
FEATURE_COLUMNS = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6",
    "bmi_age_interaction",
    "bp_bmi_interaction",
]


# Pydantic valida los datos antes de entrar en la funcion de prediccion.
class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


# Cargar el campeon una sola vez al arrancar el servicio.
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_model()
    logger.info("Modelo campeon cargado")
    yield
    logger.info("Servicio detenido")


app = FastAPI(
    title="Diabetes MLOps API",
    description="API educativa de regresion con MLflow",
    version="1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """Identificar todas las peticiones y registrar su resultado HTTP."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Peticion no controlada request_id=%s method=%s path=%s",
            request_id, request.method, request.url.path
        )
        raise

    duration_ms = (perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Peticion request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id, request.method, request.url.path,
        response.status_code, duration_ms
    )
    return response


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/model-info")
def model_info() -> dict:
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DiabetesFeatures, request: Request):
    model = get_model()

    # Aplicar exactamente las mismas transformaciones que en el entrenamiento.
    input_df = pd.DataFrame([features.model_dump()])
    input_df = create_features(input_df)
    input_df = input_df[FEATURE_COLUMNS]

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        # Registrar el motivo tecnico; al cliente solo llega un error controlado.
        logger.exception(
            "Prediccion fallida request_id=%s model=%s",
            request.state.request_id, MODEL_NAME
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction error."
        ) from exc

    result = float(prediction[0])
    info = get_model_info()

    # Trazabilidad: que se predijo, cuando, y con que version y run de MLflow.
    # No registramos las diez caracteristicas de entrada.
    logger.info(
        "Prediccion request_id=%s value=%.4f model=%s version=%s run_id=%s",
        request.state.request_id,
        result,
        info["model_name"],
        info["version"],
        info["run_id"],
    )

    return PredictionResponse(
        prediction=result,
        model_version=info["version"],
    )
