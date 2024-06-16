import hashlib

import torch
from loguru import logger


class SimpleFF(torch.nn.Module):
    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=12 * 8 * 8, out_features=12 * 8 * 4),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=12 * 8 * 4, out_features=1),
            torch.nn.Tanh(),
        )

    @logger.catch
    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        score = self.linear_relu_stack(x)
        return score

    @logger.catch
    def model_hash(self) -> str:
        """Get the hash of the model."""
        return hashlib.md5(
            (str(self.linear_relu_stack) + str(self.flatten)).encode()
        ).hexdigest()
