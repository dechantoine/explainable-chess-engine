import os
import shutil
import unittest

import chess.pgn
import numpy as np
import torch

from src.data.base_dataset import BoardItem
from src.data.parquet_dataset import ParquetChessDataset
from src.data.parquet_db import ParquetChessDB

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"


def lambda_func_board(boards: list[chess.Board]) -> list[float]:
    return [i for i in range(len(boards))]


class ParquetDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.db = ParquetChessDB(test_db_dir)
        self.db.add_directory(directory=test_data_dir,
                              funcs={"stockfish_eval": lambda_func_board})

        self.dataset = ParquetChessDataset(path=test_db_dir)

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

    def test_init(self):
        self.assertIsInstance(self.dataset, ParquetChessDataset)
        np.testing.assert_array_equal(self.dataset.indices, np.arange(1555))

    def test_len(self):
        self.assertEqual(len(self.dataset), 1555)

        self.dataset.indices = np.arange(100)
        self.assertEqual(len(self.dataset), 100)

    def test_get_hash(self):
        dataset_hash = self.dataset.get_hash()
        self.assertIsInstance(dataset_hash, str)

        self.dataset.indices = np.arange(100)
        self.assertNotEqual(dataset_hash, self.dataset.get_hash())

    def test_getitem(self):
        idx = 0
        board_item = self.dataset.__getitem__(idx)

        self.assertIsInstance(board_item, BoardItem)
        self.assertEqual(board_item.board.shape, torch.Size([12, 8, 8]))
        self.assertEqual(board_item.active_color.shape, torch.Size([1]))
        self.assertEqual(board_item.castling.shape, torch.Size([4]))
        self.assertEqual(board_item.stockfish_eval.shape, torch.Size([]))

        idx = torch.tensor(0)
        board_item = self.dataset.__getitem__(idx)

        self.assertIsInstance(board_item, BoardItem)
        self.assertEqual(board_item.board.shape, torch.Size([12, 8, 8]))
        self.assertEqual(board_item.active_color.shape, torch.Size([1]))
        self.assertEqual(board_item.castling.shape, torch.Size([4]))
        self.assertEqual(board_item.stockfish_eval.shape, torch.Size([]))

        self.dataset.indices = np.arange(100)
        idx = 100
        with self.assertRaises(IndexError):
            self.dataset.__getitem__(idx)

    def test_getitems(self):
        indices = [0, 1, 2]
        board_items = self.dataset.__getitems__(indices)

        self.assertIsInstance(board_items, BoardItem)

        self.assertEqual(board_items.board.shape, torch.Size([3, 12, 8, 8]))
        self.assertEqual(board_items.active_color.shape, torch.Size([3, 1]))
        self.assertEqual(board_items.castling.shape, torch.Size([3, 4]))
        self.assertEqual(board_items.stockfish_eval.shape, torch.Size([3]))
