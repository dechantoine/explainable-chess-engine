import os
import shutil
import unittest

import chess.pgn
import numpy as np
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
        np.testing.assert_array_equal(self.dataset.indices, np.arange(1620))

    def test_len(self):
        self.assertEqual(len(self.dataset), 1620)

        self.dataset.indices = np.arange(100)
        self.assertEqual(len(self.dataset), 100)

    def test_get_hash(self):
        dataset_hash = self.dataset.get_hash()
        self.assertIsInstance(dataset_hash, str)

        self.dataset.indices = np.arange(100)
        self.assertNotEqual(dataset_hash, self.dataset.get_hash())

    def test_getitem(self):
        idx = 0
        boards, color, castling, stockfish_eval = self.dataset.__getitem__(idx).values()

        self.assertIsInstance(boards, torch.Tensor)
        self.assertIsInstance(color, torch.Tensor)
        self.assertIsInstance(castling, torch.Tensor)
        self.assertIsInstance(stockfish_eval, torch.Tensor)
        self.assertEqual(boards.shape, torch.Size([12, 8, 8]))
        self.assertEqual(color.shape, torch.Size([1]))
        self.assertEqual(castling.shape, torch.Size([4]))
        self.assertEqual(stockfish_eval.shape, torch.Size([]))

        idx = torch.tensor(0)
        boards, color, castling, stockfish_eval = self.dataset.__getitem__(idx).values()

        self.assertIsInstance(boards, torch.Tensor)
        self.assertIsInstance(color, torch.Tensor)
        self.assertIsInstance(castling, torch.Tensor)
        self.assertIsInstance(stockfish_eval, torch.Tensor)
        self.assertEqual(boards.shape, torch.Size([12, 8, 8]))
        self.assertEqual(color.shape, torch.Size([1]))
        self.assertEqual(castling.shape, torch.Size([4]))
        self.assertEqual(stockfish_eval.shape, torch.Size([]))

        self.dataset.indices = np.arange(100)
        idx = 100
        with self.assertRaises(IndexError):
            self.dataset.__getitem__(idx)

    def test_getitems(self):
        indices = [0, 1, 2]
        boards, colors, castlings, stockfish_evals = self.dataset.__getitems__(indices).values()

        self.assertIsInstance(boards, torch.Tensor)
        self.assertIsInstance(colors, torch.Tensor)
        self.assertIsInstance(castlings, torch.Tensor)
        self.assertIsInstance(stockfish_evals, torch.Tensor)

        self.assertEqual(boards.shape, torch.Size([3, 12, 8, 8]))
        self.assertEqual(colors.shape, torch.Size([3, 1]))
        self.assertEqual(castlings.shape, torch.Size([3, 4]))
        self.assertEqual(stockfish_evals.shape, torch.Size([3]))

    def test_train_test_split(self):
        train_set, test_set = self.dataset.train_test_split(seed=42, train_size=0.8)

        self.assertEqual(len(train_set) + len(test_set), len(self.dataset))
        self.assertEqual(len(train_set), 1296)
        self.assertEqual(len(test_set), 324)

        self.assertCountEqual(
            np.concatenate([train_set.indices, test_set.indices]),
            self.dataset.indices
        )

        self.dataset.set_columns(winner=True)
        train_set, test_set = self.dataset.train_test_split(seed=42, train_size=0.8, stratify="winner")

        self.assertEqual(len(train_set) + len(test_set), len(self.dataset))
        self.assertEqual(len(train_set), 1296)
        self.assertEqual(len(test_set), 324)

        self.assertCountEqual(
            np.concatenate([train_set.indices, test_set.indices]),
            self.dataset.indices
        )

        dataset_winners = self.dataset.__getitems__(
            indices=np.arange(len(self.dataset))
            .tolist()
        )["winner"].sum().tolist()
        train_winners = train_set.__getitems__(
            indices=np.arange(len(train_set)).tolist()
        )["winner"].sum().tolist()
        test_winners = test_set.__getitems__(
            indices=np.arange(len(test_set)).tolist()
        )["winner"].sum().tolist()

        self.assertAlmostEqual(dataset_winners * 0.8, train_winners, delta=1)
        self.assertAlmostEqual(dataset_winners * 0.2, test_winners, delta=1)
