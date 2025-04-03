import pandas as pd

def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_csv(path)