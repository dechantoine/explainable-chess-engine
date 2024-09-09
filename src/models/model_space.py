import hashlib

import torch
from loguru import logger


class MultiInputConv(torch.nn.Module):
    @logger.catch
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.conv_long = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=15, padding=7, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=3, stride=2),
            torch.nn.LeakyReLU(),
        )
        self.conv_middle = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=9, padding=4, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=3, stride=2),
            torch.nn.LeakyReLU(),
        )
        self.conv_short = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, padding=2, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=3, stride=2),
            torch.nn.LeakyReLU(),
        )
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=(4 * 2 * 2) + (4 * 2 * 2) + (4 * 2 * 2) + 1 + 4, out_features=16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=16, out_features=1),
        )


    @logger.catch
    def forward(self, x):
        board, color, castling = x
        board = board.float()
        color = color.float()
        castling = castling.float()

        long = self.conv_long(board)
        long = self.flatten(long)

        middle = self.conv_middle(board)
        middle = self.flatten(middle)

        short = self.conv_short(board)
        short = self.flatten(short)

        x = torch.cat((long, middle, short, color, castling), dim=1)

        score = self.linear_relu_stack(x)
        return score

    @logger.catch
    def model_hash(self) -> str:
        """Get the hash of the model."""
        return hashlib.md5(
            (str(self.linear_relu_stack) + str(self.flatten)).encode()
        ).hexdigest()
