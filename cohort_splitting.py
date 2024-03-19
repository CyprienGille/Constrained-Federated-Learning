from typing import List
import pandas as pd


def split_into_chunks(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    chunk_size = len(df) // n_chunks
    return [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
