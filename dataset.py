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
