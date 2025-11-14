import numpy as np


def add_nan_columns(df_hour):
    df_hour["close_bs"] = np.nan
    return df_hour
