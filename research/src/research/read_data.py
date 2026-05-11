from pathlib import Path
import pandas as pd

DIFFICULTIES = ["easy","medium","hard","extrahard"]

data_dir = Path(__file__).resolve().parents[0] / "data"

def read_dataset(difficulty: str):
    return read_parquet(data_dir / f"lidar_cable_points_{difficulty}.parquet")

def read_parquet(filename: str):
    return pd.read_parquet(filename, "auto")