
import mlflow
from src.models.train import prepare_data
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient

REGISTERED_MODEL_NAME = "DiabetesRegressionModel"
MODEL_VERSION = "1"
EXPERIMENT_NAME = "Diabetes_Regression"


def passes_quality_gate(result, X_train, X_test, y_train, y_test):

    # 1. Crear el modelo de referencia
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    # 2. Calcular el error del baseline
    predicciones_baseline = baseline.predict(X_test)

    rmse_baseline = mean_squared_error(
        y_test, predicciones_baseline
    ) ** 0.5

    # 3. Recuperar el error del modelo registrado
    rmse_modelo = result.metrics["root_mean_squared_error"]

    # 4. Comparar ambos modelos
    aprobado = rmse_modelo < rmse_baseline

    # 5. Guardar el resultado en MLflow
    mlflow.log_metric("rmse_baseline", rmse_baseline)
    mlflow.log_metric("rmse_champion", rmse_modelo)

    mlflow.set_tag(
        "quality_gate",
        "passed" if aprobado else "failed"
    )

    print("RMSE baseline:", round(rmse_baseline, 2))
    print("RMSE campeón:", round(rmse_modelo, 2))
    print("Quality Gate superado:", aprobado)

    return aprobado


def assign_champion_alias(client, model_name, model_version):

    # Asignar el alias al modelo aprobado
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version
    )

    # Documentar el tipo de aprobación
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="validation_status",
        value="passed_educational_gate"
    )

    print("Alias champion asignado a la versión:", model_version)


def evaluate_registered_model():
    # 1. Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Localizar el modelo registrado
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"

    # 3. Preparar los datos reservados para prueba
    X_train, X_test, y_train, y_test = prepare_data()

    eval_data = X_test.copy()
    eval_data["target"] = y_test.values

    # 4. Crear una ejecución independiente de evaluación
    with mlflow.start_run(run_name="evaluation_v1"):

        mlflow.set_tags({
            "stage": "evaluation",
            "model_version": MODEL_VERSION,
            "registered_model_name": REGISTERED_MODEL_NAME
        })

        # 5. Evaluar con MLflow utilizando la estructura de configuración correcta
        result = mlflow.models.evaluate(
            model=model_uri,
            data=eval_data,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
            evaluator_config={"default": {"log_explainer": False}}
        )

        # 6. Mostrar resultados
        print("Modelo:", REGISTERED_MODEL_NAME)
        print("Versión:", MODEL_VERSION)
        print("Pacientes evaluados:", len(X_test))

        for metrica, valor in result.metrics.items():
            print(metrica, ":", round(valor, 4))
        
        
        # 7. Comprobar la puerta de calidad
        aprobado = passes_quality_gate(
            result,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("Resultado final:", "APROBADO" if aprobado else "RECHAZADO")


        # 8. Promocionar únicamente si supera el Quality Gate
        if aprobado:
            client = MlflowClient()

            assign_champion_alias(
                client,
                REGISTERED_MODEL_NAME,
                MODEL_VERSION
            )
        else:
            print("Modelo no promocionado: Quality Gate rechazado")



    return result


if __name__ == "__main__":
    evaluate_registered_model()
