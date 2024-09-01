import hashlib

import torch
from loguru import logger


class MultiInputFF(torch.nn.Module):
    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=(12 * 8 * 8) + 1 + 4, out_features=200),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=200, out_features=20),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=20, out_features=1),
        )

    @logger.catch
    def forward(self, x):
        board, color, castling = x
        board = board.float()
        color = color.float()
        castling = castling.float()
        board = self.flatten(board)
        x = torch.cat((board, color, castling), dim=1)
        score = self.linear_relu_stack(x)
        return score

    @logger.catch
    def model_hash(self) -> str:
        """Get the hash of the model."""
        return hashlib.md5(
            (str(self.linear_relu_stack) + str(self.flatten)).encode()
        ).hexdigest()
