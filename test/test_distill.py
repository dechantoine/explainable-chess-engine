import os
import shutil
import unittest

import chess.pgn
import numpy as np
import torch

from src.data.parquet_dataset import ParquetChessDataset
from src.data.parquet_db import ParquetChessDB
from src.train.distill import ChessEvalLoss, DistillTrainer

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"


def lambda_func_board(boards: list[chess.Board]) -> list[float]:
    return [i for i in range(len(boards))]


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=12 * 8 * 8 + 4, out_features=1)

    def forward(self, x):
        board, color, castling = x
        board = board.float()
        color = color.float()
        castling = castling.float()
        board = self.flatten(board)
        x = torch.cat((board, color, castling), dim=1)
        score = self.linear(x) * 10
        return score


class TestDistill(unittest.TestCase):
    def setUp(self):
        self.db = ParquetChessDB(test_db_dir)
        self.db.add_directory(directory=test_data_dir,
                              funcs={"stockfish_eval": lambda_func_board})

        self.dataset = ParquetChessDataset(path=test_db_dir)

        self.model = MockModel()
        self.optimizer = torch.optim.Adadelta(params=self.model.parameters(), lr=1e-3)
        self.loss = ChessEvalLoss(power=2)

        self.trainer = DistillTrainer(
            run_name="test_run",
            checkpoint_dir="test/checkpoint",
            log_dir="test/log",
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            device="cpu",
            log_sampling=0.1,
            eval_sampling=0.1,
            eval_column="stockfish_eval",
            board_column="board",
            active_color_column="active_color",
            castling_column="castling"
        )

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)
        if os.path.exists("test/checkpoint"):
            shutil.rmtree("test/checkpoint")
        if os.path.exists("test/log"):
            shutil.rmtree("test/log")

    def test_train_test_split(self):
        train_set, test_set = self.trainer.train_test_split(
            dataset=self.dataset,
            seed=42,
            train_size=0.8,
            stratify=False
        )

        self.assertEqual(len(train_set) + len(test_set), len(self.dataset))
        self.assertEqual(len(train_set), 1296)
        self.assertEqual(len(test_set), 324)

        self.assertCountEqual(
            np.concatenate([train_set.indices, test_set.indices]),
            self.dataset.indices
        )

    def test_online_validation(self):
        outputs = np.linspace(-9, 10, 100).reshape(10, 10)
        targets = np.linspace(-10, 9, 100).reshape(10, 10)

        metrics = None

        for i in range(10):
            metrics = self.trainer.online_validation(
                batch_outputs=outputs[i],
                batch_targets=targets[i],
                dict_metrics=metrics
            )

        final_online_metrics = self.trainer.validation(
            dict_metrics=metrics
        )

        final_metrics = self.trainer.validation(
            outputs=outputs.flatten(),
            targets=targets.flatten()
        )

        self.assertEqual(len(final_online_metrics), len(final_metrics))
        self.assertListEqual(list(final_online_metrics.keys()), list(final_metrics.keys()))
        np.testing.assert_array_almost_equal(np.array(list(final_online_metrics.values())),
                                             np.array(list(final_metrics.values())), decimal=6)
