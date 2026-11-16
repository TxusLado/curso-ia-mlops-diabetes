
import pandas as pd
from pydantic import BaseModel, ValidationError, ConfigDict


# 1. Definir el esquema
class DiabetesData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

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
    target: float


# 2. Validar cada fila
def validate_row(row, index):
    try:
        DiabetesData(**row)
        return True

    except ValidationError as e:
        print(f"Error in row {index}: {e}")
        return False


# 3. Validar el dataset
def validate_dataset(df):
    total = len(df)
    valid = 0
    invalid = 0

    for index, row in enumerate(df.to_dict("records")):
        if validate_row(row, index):
            valid += 1
        else:
            invalid += 1

    print("\n--- Validation Summary ---")
    print(f"Total rows: {total}")
    print(f"Valid rows: {valid}")
    print(f"Invalid rows: {invalid}")

    return invalid == 0 and total > 0


# 4. Función principal
def main():
    df = pd.read_csv("data/raw/diabetes.csv")

    if not validate_dataset(df):
        raise SystemExit(1)

    print("Data Validation Passed!")


if __name__ == "__main__":
    main()
