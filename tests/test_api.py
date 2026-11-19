import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


COLUMNAS = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6"
]


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def registro():
    datos = pd.read_csv("data/processed/diabetes_clean.csv")
    return {
        columna: float(datos.loc[0, columna])
        for columna in COLUMNAS
    }


def test_health(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200


def test_prediccion_valida(cliente, registro):
    respuesta = cliente.post("/predict", json=registro)
    assert respuesta.status_code == 200
    assert isinstance(respuesta.json()["prediction"], float)


def test_tipo_incorrecto(cliente, registro):
    registro["age"] = "hola"
    respuesta = cliente.post("/predict", json=registro)
    assert respuesta.status_code == 422


def test_campo_ausente(cliente, registro):
    del registro["bmi"]
    respuesta = cliente.post("/predict", json=registro)
    assert respuesta.status_code == 422


def test_campo_extra(cliente, registro):
    registro["nombre"] = "Don Quijote"
    respuesta = cliente.post("/predict", json=registro)
    assert respuesta.status_code == 422


def test_nan(cliente, registro):
    registro["age"] = "NaN"
    respuesta = cliente.post("/predict", json=registro)
    assert respuesta.status_code == 422


def test_model_info(cliente):
    respuesta = cliente.get("/model-info")

    assert respuesta.status_code == 200

    datos = respuesta.json()

    assert datos["model_name"] == "DiabetesRegressionModel"
    assert datos["version"] == "1"
    assert datos["alias"] == "champion"
    assert datos["run_id"]


def test_error_interno(cliente, registro):
    from unittest.mock import patch

    class ModeloQueFalla:
        def predict(self, datos):
            raise ValueError("Fallo simulado")

    with patch(
        "src.api.main.get_model",
        return_value=ModeloQueFalla()
    ):
        respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 500
    assert respuesta.json()["detail"] == "Prediction error."
