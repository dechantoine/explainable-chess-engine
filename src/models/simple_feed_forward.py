import torch
from loguru import logger

class SimpleFF(torch.nn.Module):

    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=12*8*8, out_features=12*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=12*8, out_features=12),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=12, out_features=1),
            torch.nn.Tanh()
        )

    @logger.catch
    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        score = self.linear_relu_stack(x)
        return score