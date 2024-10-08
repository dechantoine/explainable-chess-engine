import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.data.parquet_dataset import ParquetChessDataset
from src.train.base_trainer import BaseTrainer


class ChessEvalLoss(torch.nn.Module):
    def __init__(self, power: float = 2):
        """Loss function that only penalizes more the model if the signs of the outputs and targets are different."""
        super(ChessEvalLoss, self).__init__()
        self.power = power

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss
    Args:
        outputs (torch.Tensor): Outputs of the model.
        targets (torch.Tensor): Targets of the model.

    Returns:
        torch.Tensor: Loss values.

    """
        return torch.mean(
            torch.where(
                condition=torch.sign(outputs) == torch.sign(targets),
                input=torch.abs(outputs - targets) / (1 + torch.min(torch.abs(targets), torch.abs(outputs))),
                other=torch.where(condition=torch.abs(outputs - targets) > 1,
                                  input=torch.abs(outputs - targets) ** self.power,
                                  other=torch.abs(outputs - targets) ** 1 / self.power)
            )
        )

    def online_forward(self,
                       batch_outputs: torch.Tensor = None,
                       batch_targets: torch.Tensor = None,
                       current_loss: float = None,
                       n_samples: int = 0) -> torch.Tensor:
        """Forward pass of the loss computed online.

        Args:
            batch_outputs (torch.Tensor): Outputs of the model.
            batch_targets (torch.Tensor): Targets of the model.
            current_loss (float): Current loss value.
            n_samples (int): Number of samples.

        Returns:
            torch.Tensor: Loss values.

        """
        if not current_loss:
            return self(batch_outputs, batch_targets)

        new_n_samples = len(batch_outputs) + n_samples

        loss = ((current_loss * n_samples +
                 self(batch_outputs, batch_targets).item() * len(batch_outputs))
                / new_n_samples)

        return torch.Tensor([loss])


class DistillTrainer(BaseTrainer):

    def __init__(self,
                 run_name: str,
                 checkpoint_dir: str,
                 log_dir: str,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 device: torch.device,
                 log_sampling: float,
                 eval_sampling: float,
                 eval_column: str,
                 board_column: str,
                 active_color_column: str,
                 castling_column: str,
                 scheduler: torch.optim.lr_scheduler = None,
                 clip_min: float = -10,
                 clip_max: float = 10
                 ):
        """Initialize the DistillTrainer class.

        Args:
            run_name (str): Name of the run.
            checkpoint_dir (str): Directory to save checkpoints.
            log_dir (str): Directory to save logs.
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            scheduler (torch.optim.lr_scheduler): Scheduler to use.
            loss (torch.nn.Module): Loss function to use.
            device (torch.device): Device to use.
            log_sampling (float): Log every x fraction of epoch.
            eval_sampling (float): Run and log eval every x fraction of epoch.
            eval_column (str): Column to evaluate on.
            board_column (str): Column with the boards.
            active_color_column (str): Column with the active color.
            castling_column (str): Column with the castling rights.
            clip_min (float): Minimum value to clip the outputs to.
            clip_max (float): Maximum value to clip the outputs to.

        """
        super().__init__(run_name,
                         checkpoint_dir,
                         log_dir,
                         model,
                         optimizer,
                         loss,
                         device,
                         log_sampling,
                         eval_sampling,
                         clip_min,
                         clip_max)

        self.eval_column = eval_column
        self.board_column = board_column
        self.active_color_column = active_color_column
        self.castling_column = castling_column

        self.scheduler = scheduler

    def train_test_split(self,
                         dataset: ParquetChessDataset,
                         seed: int = 42,
                         train_size: float = 0.8,
                         stratify: bool = True) -> tuple[ParquetChessDataset, ParquetChessDataset]:
        """Splits the dataset into training and testing datasets.

        Args:
            dataset (ParquetChessDataset): The dataset to split.
            seed (int): The seed for reproducibility.
            train_size (float): The size of the training dataset.
            stratify (bool): Whether to stratify the dataset.

        Returns:
            tuple[ParquetChessDataset, ParquetChessDataset]: The training and testing datasets.

        """
        np.random.seed(seed)
        targets = np.zeros(len(dataset))

        if stratify:
            logger.info(f"Stratifying on {self.eval_column}.")

            targets = np.array(dataset.data.take(columns=[self.eval_column],
                                                 indices=dataset.indices.tolist())[self.eval_column],
                               ).clip(min=self.clip_min, max=self.clip_max)

            targets = np.sign(targets) * np.log1p(np.abs(targets))

            # ensure at least one pair for each targets to be able to perform stratification
            for i in np.arange(3, -1, -1):
                targets = targets.round(decimals=i)
                values, counts = np.unique(targets, return_counts=True)
                if min(counts) > 1:
                    logger.info(f"Stratification successful with {i} decimals.")
                    values = ((np.exp(np.abs(values))-1) * np.sign(values)).round(decimals=2)
                    dict_counts = dict(zip(values, counts))
                    logger.info(f"Stratification counts: {dict_counts}")
                    break

        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_indices, test_indices = next(sss.split(X=np.arange(len(dataset)), y=targets))

        train_dataset = ParquetChessDataset(path=dataset.data.path,
                                            stockfish_eval=dataset.stockfish_eval,
                                            winner=dataset.winner,
                                            move_count=dataset.move_count)
        test_dataset = ParquetChessDataset(path=dataset.data.path,
                                           stockfish_eval=dataset.stockfish_eval,
                                           winner=dataset.winner,
                                           move_count=dataset.move_count)

        train_dataset.indices = dataset.indices[train_indices]
        test_dataset.indices = dataset.indices[test_indices]

        return train_dataset, test_dataset

    def balanced_eval_signs(self,
                            dataset: ParquetChessDataset,
                            seed: int = 42) -> ParquetChessDataset:
        """Sample the dataset to have balanced evaluation signs.

        Args:
            dataset (ParquetChessDataset): The dataset to balance.
            seed (int): The seed for reproducibility.

        Returns:
            ParquetChessDataset: The balanced dataset.

        """
        eval_signs = np.array(dataset.data.take(
            indices=dataset.indices.tolist(),
            columns=["stockfish_eval"])["stockfish_eval"])
        eval_signs = np.sign(eval_signs)

        _, sign_counts = np.unique(eval_signs, return_counts=True)

        logger.info(
            f"Found {sign_counts[0]} negative samples, {sign_counts[1]} zero samples and {sign_counts[2]} positive samples.")

        max_sign = min(sign_counts[0], sign_counts[2])

        logger.info(
            f"Balancing dataset evaluation signs to {max_sign} positive samples and {max_sign} negative samples.")

        positive = np.where(eval_signs == 1)[0]
        zero = np.where(eval_signs == 0)[0]
        negative = np.where(eval_signs == -1)[0]

        np.random.seed(seed)

        if len(positive) > max_sign:
            positive = np.random.choice(positive, size=max_sign, replace=False)

        if len(negative) > max_sign:
            negative = np.random.choice(negative, size=max_sign, replace=False)

        balanced_indices = np.concatenate([positive, zero, negative])

        dataset.indices = dataset.indices[balanced_indices]
        logger.info(f"New dataset size: {len(dataset)}")

        return dataset

    def validation(self,
                   outputs: np.array = None,
                   targets: np.array = None,
                   dict_metrics: dict[str, float] = None,
                   ) -> dict[str, float]:
        """Validation metrics for the model."""
        eval_metrics = super().validation(outputs=outputs, targets=targets, dict_metrics=dict_metrics)

        if dict_metrics:
            eval_metrics["Errors/sign_error_rate"] = dict_metrics["nb_sign_error"] / dict_metrics["n_samples"]
            eval_metrics["Errors/black_sign_error_rate"] = dict_metrics["nb_black_sign_error"] / (dict_metrics[
                                                                                                      "n_samples"] -
                                                                                                  dict_metrics[
                                                                                                      "nb_white_targets"])
            eval_metrics["Errors/white_sign_error_rate"] = dict_metrics["nb_white_sign_error"] / dict_metrics[
                "nb_white_targets"]

            eval_metrics["Errors/chess_loss"] = dict_metrics["chess_loss"]

            return eval_metrics

        white_outputs = outputs[np.sign(targets) == 1]
        black_outputs = outputs[np.sign(targets) == -1]

        logger.info(f"White outputs: len : {len(white_outputs)}, values : {white_outputs}")
        logger.info(f"Black outputs: len : {len(black_outputs)}, values : {black_outputs}")

        eval_metrics["Errors/sign_error_rate"] = np.mean(np.sign(outputs) != np.sign(targets))
        eval_metrics["Errors/black_sign_error_rate"] = np.mean(black_outputs > 0)
        eval_metrics["Errors/white_sign_error_rate"] = np.mean(white_outputs < 0)

        eval_metrics["Errors/chess_loss"] = ChessEvalLoss()(torch.tensor(outputs), torch.tensor(targets)).item()

        return eval_metrics

    def online_validation(self,
                          batch_outputs: np.array,
                          batch_targets: np.array,
                          dict_metrics: dict[str, float] = None,
                          ) -> dict[str, float]:
        """Validation metrics for the model computed online."""
        super_dict_metrics = super().online_validation(batch_outputs, batch_targets, dict_metrics)

        white_outputs = batch_outputs[np.sign(batch_targets) == 1]
        black_outputs = batch_outputs[np.sign(batch_targets) == -1]

        if not dict_metrics:
            super_dict_metrics["nb_white_targets"] = np.sum(np.sign(batch_targets) == 1)
            super_dict_metrics["nb_sign_error"] = np.sum(np.sign(batch_outputs) != np.sign(batch_targets))
            super_dict_metrics["nb_black_sign_error"] = np.sum(black_outputs > 0)
            super_dict_metrics["nb_white_sign_error"] = np.sum(white_outputs < 0)
            super_dict_metrics["chess_loss"] = ChessEvalLoss().online_forward(
                batch_outputs=torch.tensor(batch_outputs),
                batch_targets=torch.tensor(batch_targets)
            ).item()

            return super_dict_metrics

        super_dict_metrics["nb_white_targets"] += np.sum(np.sign(batch_targets) == 1)
        super_dict_metrics["nb_sign_error"] += np.sum(np.sign(batch_outputs) != np.sign(batch_targets))
        super_dict_metrics["nb_black_sign_error"] += np.sum(black_outputs > 0)
        super_dict_metrics["nb_white_sign_error"] += np.sum(white_outputs < 0)
        super_dict_metrics["chess_loss"] = ChessEvalLoss().online_forward(
            batch_outputs=torch.tensor(batch_outputs),
            batch_targets=torch.tensor(batch_targets),
            current_loss=super_dict_metrics["chess_loss"],
            n_samples=super_dict_metrics["n_samples"] - len(batch_outputs)
        ).item()

        return super_dict_metrics

    def forward_validation_data(self, batch: dict[str, np.array]) -> np.array:
        """Forward a batch of validation data through the model.

        Args:
            batch (dict[str, np.array]): Batch of validation data.

        Returns:
            np.array: Outputs.

        """
        super().forward_validation_data(batch)

        boards = batch[self.board_column].to(self.device)
        active_color = batch[self.active_color_column].to(self.device)
        castling = batch[self.castling_column].to(self.device)

        with torch.no_grad():
            outputs = (self.model((boards, active_color, castling))
                       .flatten()
                       .clip(min=self.clip_min, max=self.clip_max)
                       .detach()
                       .numpy())

        return outputs

    def training_loop(self,
                      train_dataloader: torch.utils.data.DataLoader,
                      val_dataloader: torch.utils.data.DataLoader = None,
                      n_epochs: int = 1,
                      ) -> None:
        """Training loop for the model.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader.
            n_epochs (int): Number of epochs.
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader.

        """
        self.model.train()

        self.init_loop_config(len_train=len(train_dataloader))

        for epoch in tqdm(
                iterable=range(self.first_epoch, n_epochs),
                desc="Epochs",
        ):
            self.running_loss = 0.0

            for batch_idx, batch in tqdm(
                    iterable=enumerate(iterable=train_dataloader, start=self.first_batch),
                    desc="Batches",
                    total=len(train_dataloader),
            ):

                boards = (batch[self.board_column]
                          .to(self.device))
                active_color = (batch[self.active_color_column]
                                .to(self.device))
                castling = (batch[self.castling_column]
                            .to(self.device))
                targets = (batch[self.eval_column]
                           .to(self.device)
                           .clip(min=self.clip_min, max=self.clip_max))

                self.model.train()
                outputs = (self.model((boards, active_color, castling))
                           .flatten()
                           .clip(min=self.clip_min, max=self.clip_max))

                loss_value = self.training_step(
                    outputs=outputs, targets=targets
                )

                self.running_loss += loss_value

                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    self.log_train(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        len_trainset=len(train_dataloader),
                    )
                    self.running_loss = 0.0

                if batch_idx % self.eval_interval == 0 or batch_idx == len(train_dataloader) - 1:
                    #validation_outputs = self.forward_validation_data(val_dataloader=val_dataloader)

                    self.log_eval(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        val_dataloader=val_dataloader,
                        len_trainset=len(train_dataloader),
                        hparams={
                            "mode": "distill_stockfish",
                            "model": self.model.__class__.__name__,
                            "model_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                            "model_hash": self.model.model_hash(),
                            "optimizer": self.optimizer.__class__.__name__,
                            "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
                            "lr_decay": self.optimizer.state_dict()["param_groups"][0]["lr_decay"]
                            if "lr_decay" in self.optimizer.state_dict()["param_groups"][0]
                            else None,
                            "weight_decay": self.optimizer.state_dict()["param_groups"][0]["weight_decay"]
                            if "weight_decay" in self.optimizer.state_dict()["param_groups"][0]
                            else None,
                            "loss": self.loss.__class__.__name__,
                            "n_epochs": n_epochs,
                            "batch_size": train_dataloader.batch_size,
                            "train_dataset_hash": train_dataloader.dataset.get_hash(),
                            "test_dataset_hash": val_dataloader.dataset.get_hash(),
                        },
                    )

            if self.scheduler:
                self.scheduler.step()

        self.writer.close()
