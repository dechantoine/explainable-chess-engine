import os
import shutil
import sys
import unittest

import numpy as np
import torch
from _pytest.monkeypatch import MonkeyPatch

from src.data.parquet_dataset import ParquetChessDataset
from src.data.parquet_db import ParquetChessDB
from src.train.base_trainer import CHECKPOINT_PREFIX, START_FROM_SCRATCH, BaseTrainer

test_db_dir = "test/test_db"
test_data_dir = "test/test_data"
test_checkpoint_dir = "test/models_checkpoint"
test_log_dir = "test/logs"


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


class TrainUtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.monkeypatch = MonkeyPatch()

        self.db = ParquetChessDB(test_db_dir)
        self.db.add_directory(directory=test_data_dir)

        self.dataset = ParquetChessDataset(path=test_db_dir)

        self.model = MockModel()
        self.optimizer = torch.optim.Adadelta(params=self.model.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()

        self.run_name = "test_run"

        os.makedirs(os.path.join(test_checkpoint_dir, self.run_name), exist_ok=True)
        for i in range(10):
            torch.save(
                obj={
                    "model_state_dict": self.model.state_dict(),
                    "epoch": i,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": 0.1 * i},
                f=f"{test_checkpoint_dir}/{self.run_name}/{CHECKPOINT_PREFIX}{i}.pt")

        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: "10")
        self.trainer = BaseTrainer(
            run_name=self.run_name,
            checkpoint_dir=test_checkpoint_dir,
            log_dir="test/log",
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            device="cpu",
            log_sampling=0.1,
            eval_sampling=0.1,
        )


    def tearDown(self):
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)

    def test_list_existing_models(self):
        models = self.trainer.list_existing_models()

        self.assertEqual(len(models), 9)
        self.assertEqual(models, list(range(9)))

    def test_ask_user_for_checkpoint(self):
        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: "5")
        checkpoint = self.trainer.ask_user_for_checkpoint()

        self.assertEqual(checkpoint, "5")

    def test_init_training(self):
        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: "5")
        self.trainer.init_training()

        self.assertEqual(self.trainer.resume_step, 5)
        self.assertIsInstance(self.trainer.model, torch.nn.Module)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Optimizer)

        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: START_FROM_SCRATCH)
        self.trainer.init_training()

        self.assertEqual(self.trainer.resume_step, 0)
        self.assertIsInstance(self.trainer.model, torch.nn.Module)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Optimizer)


    def test_validation(self):
        metrics = self.trainer.validation(
            outputs=np.arange(-9,10),
            targets=np.arange(-10,9)
        )

        self.assertIsInstance(metrics, dict)