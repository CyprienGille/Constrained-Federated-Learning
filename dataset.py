from typing import Optional, List
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ArtificialDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X.astype("float32")
        self.Y = y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.Y)


def get_train_test_data(
    train_prop: float,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_repeated: int,
    n_clusters_per_class: int,
    weights: Optional[List[float]],
    class_sep: float,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=n_clusters_per_class,
        n_classes=2,
        weights=weights,
        class_sep=class_sep,
        hypercube=True,
        shuffle=True,
        scale=1.0,
    )

    # Split into two cohorts
    idx_1, idx_2 = train_test_split(arange(n_samples), train_size=0.5)
    X_1, X_2 = X[idx_1], X[idx_2]
    y_1, y_2 = y[idx_1], y[idx_2]

    # Then split into training and testing sets
    train_idx_1, test_idx_1 = train_test_split(arange(len(y_1)), train_size=train_prop)
    X_train_1, X_test_1 = X_1[train_idx_1], X_1[test_idx_1]
    y_train_1, y_test_1 = y_1[train_idx_1], y_1[test_idx_1]

    train_idx_2, test_idx_2 = train_test_split(arange(len(y_2)), train_size=train_prop)
    X_train_2, X_test_2 = X_2[train_idx_2], X_2[test_idx_2]
    y_train_2, y_test_2 = y_2[train_idx_2], y_2[test_idx_2]

    return (
        X_train_1,
        y_train_1,
        X_test_1,
        y_test_1,
        X_train_2,
        y_train_2,
        X_test_2,
        y_test_2,
    )
