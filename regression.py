import torch
import torch.nn.functional as f
torch.manual_seed(1)
torch.cuda.set_device(0)


class Regression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Regression, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x), inplace=True)
        x = self.predict(x)
        return x


