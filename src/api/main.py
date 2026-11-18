from contextlib import asynccontextmanager
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.api.model import (
    MODEL_NAME,
    load_champion_model,
    get_model,
    get_model_info,
)
from src.data.features import create_features


logger = logging.getLogger(__name__)

# Columnas que espera nuestro modelo campeón
FEATURE_COLUMNS = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6",
    "bmi_age_interaction",
    "bp_bmi_interaction",
]


# Datos recibidos por la API
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


# Respuesta de una predicción
class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


# Cargar el campeón una sola vez al arrancar
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_model()
    logger.info("Modelo campeón cargado")
    yield
    logger.info("Servicio detenido")


app = FastAPI(
    title="Diabetes MLOps API",
    description="API educativa de regresión con MLflow",
    version="1.0",
    lifespan=lifespan,
)


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
def predict(features: DiabetesFeatures):
    model = get_model()

    # Convertir los 10 datos recibidos a DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    # Aplicar las mismas transformaciones del entrenamiento
    input_df = create_features(input_df)

    # Respetar el contrato del modelo: 12 columnas
    input_df = input_df[FEATURE_COLUMNS]

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        logger.exception("Error al realizar la predicción")
        raise HTTPException(
            status_code=500,
            detail="Prediction error."
        ) from exc

    result = float(prediction[0])

    logger.info("Predicción realizada: %.4f", result)

    return PredictionResponse(
        prediction=result,
        model_version=get_model_info()["version"],
    )
