from src.data.dataset import ChessBoardDataset
from src.train.train_utils import train_test_split

from loguru import logger

import unittest


class TrainUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = ChessBoardDataset(root_dir="../test/test_data",
                                         return_moves=False,
                                         return_outcome=False,
                                         transform=False)

    @logger.catch
    def test_train_test_split(self):
        train_set, test_set = train_test_split(self.dataset, seed=42, train_size=0.8)

        assert train_set.list_pgn_files == test_set.list_pgn_files
        assert isinstance(train_set, ChessBoardDataset)
        assert isinstance(test_set, ChessBoardDataset)
        assert len(train_set) + len(test_set) == len(self.dataset)
        assert all(i not in test_set.board_indices for i in train_set.board_indices)
        assert all(i not in train_set.board_indices for i in test_set.board_indices)
        self.assertCountEqual(train_set.board_indices + test_set.board_indices,
                              self.dataset.board_indices)
