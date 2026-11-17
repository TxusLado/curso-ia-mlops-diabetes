
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from src.data.features import create_features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Preparar los datos
def prepare_data():
    df = pd.read_csv("data/processed/diabetes_clean.csv")

    df_eng = create_features(df)

    X = df_eng.drop(columns=['target'])
    y = df_eng['target']

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


# 2. Función principal
def main():
    X_train, X_test, y_train, y_test = prepare_data()

    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    print("Modelos candidatos:", list(modelos.keys()))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes_Regression")


    for nombre, modelo in modelos.items():

        with mlflow.start_run(run_name=f"{nombre}_CV"):

            resultados = cross_val_score(
                modelo,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error"
            )

            rmse_cv = -resultados.mean()

            mlflow.log_param("modelo", nombre)
            mlflow.log_param("features", X_train.shape[1])
            mlflow.log_param("n_splits", 5)

            mlflow.log_metric("rmse_cv", rmse_cv)

            print(nombre, "RMSE CV:", round(rmse_cv, 2))


if __name__ == "__main__":
    main()
