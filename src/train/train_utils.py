from src.data.dataset import ChessBoardDataset

from loguru import logger
import numpy as np
import torch
from copy import deepcopy


@logger.catch
def train_test_split(dataset: ChessBoardDataset, seed: int, train_size: float) -> (
ChessBoardDataset, ChessBoardDataset):
    """Split the provided dataset into a training and testing set.

    Args:
        dataset (ChessBoardDataset): Dataset to split.
        seed (int): Seed for the random split.
        train_size (float): Proportion of the training set.

    Returns:
        ChessBoardDataset: Training dataset.
        ChessBoardDataset: Testing dataset.
    """
    np.random.seed(seed)

    indices = np.random.permutation(len(dataset))
    split = int(train_size * len(dataset))
    train_indices, test_indices = indices[:split], indices[split:]

    train_set = deepcopy(dataset)
    test_set = deepcopy(dataset)

    train_set.board_indices = [dataset.board_indices[i] for i in train_indices]
    test_set.board_indices = [dataset.board_indices[i] for i in test_indices]

    return train_set, test_set


@logger.catch
def reward_fn(outcome: torch.Tensor, gamma: float = 0.99):
    """Calculate the reward for the given outcome.

    Args:
        outcome (torch.Tensor): Outcome of the game.
        gamma (float): Discount factor.

    Returns:
        float: Reward for the given outcome.
    """
    return (gamma ** (outcome[:, 1] - outcome[:, 0])) * outcome[:, 2]
