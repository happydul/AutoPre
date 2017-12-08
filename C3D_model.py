import torch.nn as nn
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(9, 1, kernel_size=(1, 2, 2), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))
        self.fc8 = nn.Linear(1728, 192)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.dropout(h)
        h = h.view(-1, 8, 8)
        return h

