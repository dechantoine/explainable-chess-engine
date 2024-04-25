import unittest

import torch

from src.data.dataset import ChessBoardDataset
from src.train.train_utils import reward_fn, train_test_split


class TrainUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=False,
            return_outcome=False,
            transform=False,
        )

        self.outcomes = torch.tensor([[50, 60, 1], [20, 40, 0], [10, 40, -1]])
        self.gamma = 0.99
        self.expected = torch.tensor([0.99**10, 0, -1 * (0.99**30)])

    def test_train_test_split(self):
        train_set, test_set = train_test_split(self.dataset, seed=42, train_size=0.8)

        assert train_set.list_pgn_files == test_set.list_pgn_files
        assert isinstance(train_set, ChessBoardDataset)
        assert isinstance(test_set, ChessBoardDataset)
        assert len(train_set) + len(test_set) == len(self.dataset)
        assert all(i not in test_set.board_indices for i in train_set.board_indices)
        assert all(i not in train_set.board_indices for i in test_set.board_indices)
        self.assertCountEqual(
            train_set.board_indices + test_set.board_indices, self.dataset.board_indices
        )

    def test_reward_fn(self):
        rewards = reward_fn(outcome=self.outcomes, gamma=0.99)

        torch.testing.assert_allclose(rewards, self.expected)
