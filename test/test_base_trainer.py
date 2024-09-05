import os
import shutil
import sys
import unittest
from test.mock_torch_model import MockModel

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
