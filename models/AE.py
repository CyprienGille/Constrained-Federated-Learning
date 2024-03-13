from torch import nn


class netBio(nn.Module):
    def __init__(self, n_inputs, n_outputs=2, n_hidden=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_outputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_inputs),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
