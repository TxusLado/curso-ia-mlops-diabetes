
import pandas as pd
from sklearn.datasets import load_diabetes
from pathlib import Path

# 1. Cargar los datos
diabetes = load_diabetes()

#2. Crear dataframe
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

#3. Crear la carpeta de destino
Path("data/raw").mkdir(parents=True, exist_ok=True)

#4. Guardar el archivo
df.to_csv("data/raw/diabetes.csv", index= False)

print("Extracción completada")
print (df.shape)
