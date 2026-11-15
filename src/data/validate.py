
import pandas as pd
import numpy as np

# Load the data

df = pd.read_csv("data/raw/diabetes.csv")

print(df.shape)

assert 'target' in df.columns, "Target column missing!"

assert len(df) > 0, "Dataset is empty!"

assert df.select_dtypes(include=[np.number]).shape[1] == df.shape[1], "Non-numeric data detected!"

assert df.isnull().sum().sum() == 0, "Critical missing values found!"



columnas = ['bmi', 'age', 'bp']
for columna in columnas:
    assert columna in df.columns, f" {columna} is missing"

print("Data Validation Passed!")

