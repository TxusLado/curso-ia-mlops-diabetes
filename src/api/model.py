from pathlib import Path
import mlflow
from mlflow import MlflowClient
from fastapi import HTTPException

# Configuración de MLflow
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")

MODEL_NAME = "DiabetesRegressionModel"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Estado interno: modelo y metadatos conservados entre peticiones
_state = {
    "model": None,
    "model_version": None,
    "run_id": None,
}


def load_champion_model() -> None:
    """Cargar el campeón desde MLflow al iniciar FastAPI."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )

    _state["model_version"] = str(mv.version)
    _state["run_id"] = mv.run_id


def get_model():
    """Devolver el modelo cargado en memoria."""
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    return _state["model"]


def get_model_info() -> dict:
    """Devolver información de la versión cargada."""
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    return {
        "model_name": MODEL_NAME,
        "version": _state["model_version"],
        "alias": MODEL_ALIAS,
        "run_id": _state["run_id"],
    }
