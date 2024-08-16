import os
import shutil
import sys
import unittest

import numpy as np
import torch
from _pytest.monkeypatch import MonkeyPatch

from src.data.pgn_dataset import PGNDataset
from src.train.train import collate_fn
from src.train.train_utils import (
    CHECKPOINT_PREFIX,
    START_FROM_SCRATCH,
    ask_user_for_checkpoint,
    gradient_descent_values,
    init_training,
    list_existing_models,
    reward_fn,
    train_test_split,
    training_step,
    validation,
    validation_values,
)

test_data_dir = "test/test_data"
test_checkpoint_dir = "test/models_checkpoint"
test_log_dir = "test/logs"


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=12 * 8 * 8, out_features=1)

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        return self.linear(x)


class TrainUtilsTestCase(unittest.TestCase):

    def setUp(self):
        dataset = PGNDataset(
            root_dir=test_data_dir,
            return_moves=False,
            return_outcome=True,
            transform=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
        )

        self.outcomes = torch.tensor([[50, 60, 1], [20, 40, 0], [10, 40, -1]])
        self.gamma = 0.99
        self.expected = torch.tensor([0.99 ** 10, 0, -1 * (0.99 ** 30)])

        self.dataset = PGNDataset(
            root_dir=test_data_dir,
            return_moves=False,
            return_outcome=True,
            transform=False,
            include_draws=False,
            in_memory=True,
            num_workers=8,
        )

        self.run_name = "dummy_run"
        self.dummy_model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.dummy_model.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss()
        os.makedirs(os.path.join(test_checkpoint_dir, self.run_name), exist_ok=True)
        for i in range(10):
            torch.save(
                obj={
                    "model_state_dict": self.dummy_model.state_dict(),
                    "epoch": i,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": 0.1 * i},
                f=f"{test_checkpoint_dir}/{self.run_name}/{CHECKPOINT_PREFIX}{i}.pt")

        self.monkeypatch = MonkeyPatch()

        self.model = MockModel()

        self.outputs = np.array([[0.1], [0.2], [0.3]])
        self.targets = np.array([[-0.1], [0.2], [0.5]])

    def tearDown(self):
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)

    def test_train_test_split(self):
        train_set, test_set = train_test_split(self.dataset, seed=42, train_size=0.8)

        assert train_set.list_pgn_files == test_set.list_pgn_files
        assert isinstance(train_set, PGNDataset)
        assert isinstance(test_set, PGNDataset)
        assert len(train_set) + len(test_set) == len(self.dataset)
        assert all(i not in test_set.board_indices for i in train_set.board_indices)
        assert all(i not in train_set.board_indices for i in test_set.board_indices)
        self.assertCountEqual(
            train_set.board_indices + test_set.board_indices, self.dataset.board_indices
        )

    def test_reward_fn(self):
        rewards = reward_fn(outcome=self.outcomes, gamma=0.99)

        torch.testing.assert_allclose(rewards, self.expected)

    def test_list_existing_models(self):
        models = list_existing_models(run_name=self.run_name, checkpoint_dir=test_checkpoint_dir)

        assert len(models) == 10
        assert models == list(range(10))

    def test_ask_user_for_checkpoint(self):
        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: "5")
        checkpoint = ask_user_for_checkpoint(run_name=self.run_name, checkpoint_dir=test_checkpoint_dir)

        assert checkpoint == "5"

    def test_init_training(self):
        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: "5")
        model, optimizer, resume_step = init_training(
            run_name=self.run_name,
            checkpoint_dir=test_checkpoint_dir,
            log_dir=test_log_dir,
            model=self.dummy_model,
            optimizer=self.optimizer
        )

        assert resume_step == 5
        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)

        self.monkeypatch.setattr(target=sys.stdin, name="readline", value=lambda: START_FROM_SCRATCH)
        model, optimizer, resume_step = init_training(
            run_name=self.run_name,
            checkpoint_dir=test_checkpoint_dir,
            log_dir=test_log_dir,
            model=self.dummy_model,
            optimizer=self.optimizer
        )

        assert resume_step == 0
        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_validation_values(self):
        outputs, targets = validation_values(
            model=self.model,
            test_dataloader=self.test_dataloader,
            gamma=self.gamma,
            device="cpu"
        )

        assert isinstance(outputs, np.ndarray)
        assert isinstance(targets, np.ndarray)

    def test_validation(self):
        metrics = validation(
            outputs=self.outputs,
            targets=self.targets
        )

        assert isinstance(metrics, dict)

    def test_gradient_descent_values(self):
        boards, outcomes = next(iter(self.test_dataloader))

        outputs, targets = gradient_descent_values(
            model=self.model,
            boards=boards,
            outcomes=outcomes,
            gamma=self.gamma,
        )

        assert isinstance(outputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

    def test_training_step(self):
        boards, outcomes = next(iter(self.test_dataloader))

        outputs, targets = gradient_descent_values(
            model=self.model,
            boards=boards,
            outcomes=outcomes,
            gamma=self.gamma,
        )

        loss = training_step(
            optimizer=self.optimizer,
            loss=self.loss,
            outputs=outputs,
            targets=targets,
        )

        assert isinstance(loss, float)
