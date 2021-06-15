import torch.nn as nn


class MiniModel(nn.Module):

    def __init__(self):
        super(MiniModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 896),
            nn.Linear(896, 1024)
        )

    def forward(self, x):
        return self.network(x)
