import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

N_SAMPLES = 10_000
N_FEATURES = 300
N_INFORMATIVE = 20
N_REDUNDANT = 0
N_REPEATED = 0
WEIGHTS = None
# WEIGHTS = [0.9, 0.1] # To make an unbalanced dataset
CLASS_SEP = 1.0

data_dir = "data/"

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=N_REPEATED,
        weights=WEIGHTS,
        class_sep=CLASS_SEP,
        n_clusters_per_class=1,
        n_classes=2,
        hypercube=True,
        shuffle=True,
        scale=1.0,
    )

    df_X = pd.DataFrame(
        index=[f"Sample_{i}" for i in range(N_SAMPLES)],
        columns=[f"Feature_{i}" for i in range(N_FEATURES)],
        data=X,
    )

    df_y = pd.DataFrame(
        index=[f"Sample_{i}" for i in range(N_SAMPLES)],
        columns=["Label"],
        data=y,
    )

    data_df = pd.concat([df_y, df_X], axis=1)
    data_df.to_csv(
        f"{data_dir}Synth_{N_FEATURES}f_{N_INFORMATIVE}inf_{N_SAMPLES}s.csv",
        index_label="Name",
        sep=";",
    )
