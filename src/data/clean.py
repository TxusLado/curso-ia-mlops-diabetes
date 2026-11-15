
import pandas as pd
import numpy as np
from pathlib import Path

# Cargar datos originales
df = pd.read_csv("data/raw/diabetes.csv")

# Función de limpieza
def clean_data(input_df):
    cleaned_df = input_df.copy()
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.drop_duplicates()

    for col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col])

    return cleaned_df

# Limpiar los datos
df_cleaned = clean_data(df)

print("Antes: ",df.shape)
print("Después: ",df_cleaned.shape)

Path("data/processed").mkdir(parents=True, exist_ok=True)

df_cleaned.to_csv("data/processed/diabetes_clean.csv", index=False)

print("Limpieza completada")
