import pandas as pd

def analyze_data(df: pd.DataFrame):
    summary = {
        "tipos": df.dtypes.to_dict(),
        "describe": df.describe(include='all').to_dict(),
        "shape": df.shape
    }
    return summary
