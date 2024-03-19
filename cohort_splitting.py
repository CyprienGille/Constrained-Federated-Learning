from typing import List
import pandas as pd


def split_into_chunks(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    chunk_size = (len(df) // n_chunks) + 1
    return [
        df.iloc[i : i + chunk_size].copy().reset_index(drop=True)
        for i in range(0, len(df), chunk_size)
    ]
