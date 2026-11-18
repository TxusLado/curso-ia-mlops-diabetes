
import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "Diabetes_Regression"


def get_best_run(client, experiment_name):
    # Buscar el experimento
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError("Experimento no encontrado")

    # Ordenar las ejecuciones por RMSE CV
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.rmse_cv ASC"]
    )

    # Seleccionar la primera ejecución con modelo guardado
    for run in runs:
        if "rmse_cv" in run.data.metrics and run.data.tags.get("model_uri"):
            return run

    raise ValueError("No hay modelos válidos registrados")

# Nombre del modelo en el registro
REGISTERED_MODEL_NAME = "DiabetesRegressionModel"


def register_model(client, best_run):
    # Recuperar el modelo previamente guardado
    model_uri = best_run.data.tags["model_uri"]

    # Crear una versión en el Model Registry
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    # Pendiente de evaluación final
    client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="validation_status",
        value="pending"
    )

    return model_version


def add_model_description(client, best_run, model_version):
    # Recuperar información del experimento
    metrics = best_run.data.metrics
    params = best_run.data.params

    # Preparar la descripción del modelo
    description = (
        f"Modelo seleccionado: {params['modelo']}\n"
        f"RMSE CV (5 folds): {metrics['rmse_cv']:.2f}\n"
        f"Características: {params['features']}\n"
        f"Pacientes de entrenamiento: 353\n"
        f"Pacientes reservados para prueba: 89\n"
        f"Run ID: {best_run.info.run_id}\n"
        f"Estado: pendiente de evaluación final."
    )

    # Actualizar la descripción de la versión registrada
    client.update_model_version(
        name=model_version.name,
        version=model_version.version,
        description=description
    )

    print("Descripción del modelo actualizada")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    best_run = get_best_run(client, EXPERIMENT_NAME)

    print("Campeón:", best_run.data.params["modelo"])
    print("RMSE CV:", round(best_run.data.metrics["rmse_cv"], 2))
    print("URI:", best_run.data.tags["model_uri"])


    model_version = register_model(client, best_run)

    print("Modelo registrado:", model_version.name)
    print("Versión:", model_version.version)



if __name__ == "__main__":
    main()
