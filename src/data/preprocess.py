
import pandas as pd
from pathlib import Path
from src.data.features import create_features


def audit_outliers(df):
    # Añadir las nuevas características
    df_eng = create_features(df)
    X = df_eng.drop(columns=['target'])

    # Calcular umbrales estadísticos
    p1 = X.quantile(0.01)
    p99 = X.quantile(0.99)

    # Identificar valores extremos
    mask = (X < p1) | (X > p99)

    # Crear un informe por característica
    report = pd.DataFrame({
        'feature': X.columns,
        'p1': p1.values,
        'p99': p99.values,
        'outliers': mask.sum().values
    })

    affected = mask.any(axis=1).sum()

    return report, affected


def main():
    df = pd.read_csv("data/processed/diabetes_clean.csv")

    report, affected = audit_outliers(df)

    # Guardar el informe
    Path("reports").mkdir(parents=True, exist_ok=True)
    report.to_csv("reports/outliers.csv", index=False)

    print("Total de pacientes:", len(df))
    print("Pacientes con valores extremos:", affected)
    print("Porcentaje:", round(affected / len(df) * 100, 2), "%")
    print("Decisión: conservar todos los registros.")
    print("Informe guardado en reports/outliers.csv")


if __name__ == "__main__":
    main()
