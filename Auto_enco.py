import torch.nn as nn


class MyAuto(nn.Module):
    def __init__(self):
        super(MyAuto, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(36864, 4096),
            nn.Tanh(),
            nn.Linear(4096, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 1024),
            nn.Tanh(),
            nn.Linear(1024, 4096),
            nn.Tanh(),
            nn.Linear(4096, 36864),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        re_encoded = encoded.view(-1, 8, 8)
        return re_encoded, decoded
