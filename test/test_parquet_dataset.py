import os
import shutil
import unittest

import chess.pgn
import torch

from src.data.parquet_dataset import ParquetChessDataset
from src.data.parquet_db import ParquetChessDB

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"


def lambda_func_board(boards: list[chess.Board]) -> list[float]:
    return [i for i in range(len(boards))]


class TestParquetChessDB(unittest.TestCase):
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

    def test_len(self):
        self.assertEqual(len(self.dataset), 1620)

    def test_getitem(self):
        idx = 0
        indexes, color, castling, stockfish_eval = self.dataset.__getitem__(idx)

        self.assertIsInstance(indexes, torch.Tensor)
        self.assertIsInstance(color, torch.Tensor)
        self.assertIsInstance(castling, torch.Tensor)
        self.assertIsInstance(stockfish_eval, torch.Tensor)
        self.assertEqual(indexes.shape, torch.Size([12, 8, 8]))
        self.assertEqual(color.shape, torch.Size([]))
        self.assertEqual(castling.shape, torch.Size([4]))
        self.assertEqual(stockfish_eval.shape, torch.Size([]))

        idx = torch.tensor(0)
        indexes, color, castling, stockfish_eval = self.dataset.__getitem__(idx)

        self.assertIsInstance(indexes, torch.Tensor)
        self.assertIsInstance(color, torch.Tensor)
        self.assertIsInstance(castling, torch.Tensor)
        self.assertIsInstance(stockfish_eval, torch.Tensor)
        self.assertEqual(indexes.shape, torch.Size([12, 8, 8]))
        self.assertEqual(color.shape, torch.Size([]))
        self.assertEqual(castling.shape, torch.Size([4]))
        self.assertEqual(stockfish_eval.shape, torch.Size([]))

    def test_getitems(self):
        indices = [0, 1, 2]
        indexes, colors, castlings, stockfish_evals = self.dataset.__getitems__(indices)

        self.assertIsInstance(indexes, torch.Tensor)
        self.assertIsInstance(colors, torch.Tensor)
        self.assertIsInstance(castlings, torch.Tensor)
        self.assertIsInstance(stockfish_evals, torch.Tensor)

        self.assertEqual(indexes.shape, torch.Size([3, 12, 8, 8]))
        self.assertEqual(colors.shape, torch.Size([3]))
        self.assertEqual(castlings.shape, torch.Size([3, 4]))
        self.assertEqual(stockfish_evals.shape, torch.Size([3]))
