import hashlib

import torch
from loguru import logger


class MultiInputConv(torch.nn.Module):
    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=64, kernel_size=15, padding=7, stride=1),
            torch.nn.LeakyReLU(),
        )
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=(64 * 8 * 8) + 1 + 4, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=16, out_features=1),
        )


    @logger.catch
    def forward(self, x):
        board, color, castling = x
        board = board.float()
        color = color.float()
        castling = castling.float()

        conv = self.conv(board)
        conv = self.flatten(conv)

        x = torch.cat((conv, color, castling), dim=1)

        score = self.linear_relu_stack(x)
        return score

    @logger.catch
    def model_hash(self) -> str:
        """Get the hash of the model."""
        return hashlib.md5(
            (str(self.linear_relu_stack) + str(self.flatten)).encode()
        ).hexdigest()
