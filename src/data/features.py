
import pandas as pd

def create_features(df):
    df_eng = df.copy()

    df_eng['bmi_age_interaction'] = df_eng['bmi'] * df_eng['age']
    df_eng['bp_bmi_interaction'] = df_eng['bp'] * df_eng['bmi']
#    df_eng['s1_s2_ratio'] = df_eng['s1'] / (df_eng['s2'] + 1e-9)

    return df_eng

