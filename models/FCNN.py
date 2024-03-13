from torch import nn


class FCNN(nn.Module):
    def __init__(self, n_inputs: int, n_outputs=2, n_hidden=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x):
        return self.encoder(x)
