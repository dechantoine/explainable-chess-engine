import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.data.parquet_dataset import ParquetChessDataset
from src.train.base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):

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
                 board_column: str,
                 active_color_column: str,
                 castling_column: str,
                 winner_column: str,
                 total_moves_column: str,
                 game_len_column: str,
                 gamma: float,
                 ):
        """Initialize the DistillTrainer class.

        Args:
            run_name (str): Name of the run.
            checkpoint_dir (str): Directory to save checkpoints.
            log_dir (str): Directory to save logs.
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            loss (torch.nn.Module): Loss function to use.
            device (torch.device): Device to use.
            log_sampling (float): Log every x fraction of epoch.
            eval_sampling (float): Run and log eval every x fraction of epoch.
            board_column (str): Column name for the board.
            active_color_column (str): Column name for the active color.
            castling_column (str): Column name for the castling.
            winner_column (str): Column name for the winner.
            total_moves_column (str): Column name for the total moves.
            game_len_column (str): Column name for the game length.
            gamma (float): Discount factor.

        """
        super().__init__(run_name, checkpoint_dir, log_dir, model, optimizer, loss, device, log_sampling, eval_sampling)

        self.gamma = gamma

        self.board_column = board_column
        self.active_color_column = active_color_column
        self.castling_column = castling_column
        self.winner_column = winner_column
        self.total_moves_column = total_moves_column
        self.game_len_column = game_len_column

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
            stratify (bool): Whether to stratify the split.

        Returns:
            tuple[ParquetChessDataset, ParquetChessDataset]: The training and testing datasets.

        """
        np.random.seed(seed)
        targets = np.zeros(len(dataset))

        if stratify:
            logger.info("Stratifying on discount factor.")

            targets = self.gamma_discounted_targets(
                total_moves=dataset.data.take(columns=[self.total_moves_column],
                                              indices=dataset.indices.tolist())[self.total_moves_column],
                winner=dataset.data.take(columns=[self.winner_column],
                                         indices=dataset.indices.tolist())[self.winner_column],
                active_color=dataset.data.take(columns=[self.active_color_column],
                                               indices=dataset.indices.tolist())[self.active_color_column],
                game_len=dataset.data.take(columns=[self.game_len_column],
                                           indices=dataset.indices.tolist())[self.game_len_column]
            ).numpy()

            # ensure at least one pair for each targets to be able to perform stratification
            for i in np.arange(3, -1, -1):
                targets = targets.round(decimals=i)
                values, counts = np.unique(targets, return_counts=True)
                if min(counts) > 1:
                    logger.info(f"Stratification successful with {i} decimals.")
                    dict_counts = dict(zip(values, counts))
                    logger.info(f"Stratification counts: {dict_counts}")
                    break

        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_indices, test_indices = next(sss.split(X=np.arange(len(dataset)), y=targets))

        train_dataset = ParquetChessDataset(path=dataset.data.path,
                                            stockfish_eval=dataset.stockfish_eval,
                                            winner=dataset.winner)
        test_dataset = ParquetChessDataset(path=dataset.data.path,
                                           stockfish_eval=dataset.stockfish_eval,
                                           winner=dataset.winner)

        train_dataset.indices = dataset.indices[train_indices]
        test_dataset.indices = dataset.indices[test_indices]

        return train_dataset, test_dataset

    def validation(self,
                   outputs: np.array,
                   targets: np.array,
                   ) -> dict[str, dict[str, float]]:
        """Validation metrics for the model."""
        eval_scalars = super().validation(outputs, targets)

        white_outputs = outputs[np.sign(targets) == 1]
        black_outputs = outputs[np.sign(targets) == -1]

        eval_scalars["Errors/sign_error_rate"] = np.mean(np.sign(outputs) != np.sign(targets))
        eval_scalars["Errors/black_sign_error_rate"] = np.mean(black_outputs > 0)
        eval_scalars["Errors/white_sign_error_rate"] = np.mean(white_outputs < 0)

        eval_scalars["Errors/chess_loss"] = self.loss(torch.tensor(outputs), torch.tensor(targets)).item()

        return eval_scalars

    def forward_validation_data(self, val_dataloader: torch.utils.data.DataLoader) -> np.array:
        """Forward the validation data through the model.

        Args:
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader.

        Returns:
            np.array: Validation outputs.

        """
        super().forward_validation_data(val_dataloader)

        validation_outputs = []
        for i, val_batch in enumerate(val_dataloader):
            boards = val_batch[self.board_column].to(self.device)
            active_color = val_batch[self.active_color_column].to(self.device)
            castling = val_batch[self.castling_column].to(self.device)
            outputs = self.model((boards, active_color, castling)).detach()
            validation_outputs.extend(outputs)

        validation_outputs = (torch.stack(validation_outputs)
                              .flatten()
                              .detach()
                              .numpy())

        return validation_outputs

    def gamma_discounted_targets(self,
                                 total_moves: torch.Tensor,
                                 winner: torch.Tensor,
                                 active_color: torch.Tensor,
                                 game_len: torch.Tensor) -> torch.Tensor:
        """Calculate the gamma discounted targets.

        Args:
            total_moves (torch.Tensor): Total moves.
            winner (torch.Tensor): Winner.
            active_color (torch.Tensor): Active color.
            game_len (torch.Tensor): Game length.

        Returns:
            torch.Tensor: Gamma discounted targets.

        """
        # torch.Tensor((gamma ** (outcome[:, 1] - outcome[:, 0])) * outcome[:, 2])
        moves_to_end = game_len - (total_moves * 2 - active_color)
        targets = winner * torch.pow(self.gamma, moves_to_end)
        return targets

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

        validation_targets = []
        for i, batch in enumerate(val_dataloader):
            batch_targets = self.gamma_discounted_targets(
                total_moves=batch[self.total_moves_column],
                winner=batch[self.winner_column],
                active_color=batch[self.active_color_column],
                game_len=batch[self.game_len_column]
            )
            validation_targets.extend(batch_targets)

        validation_targets = (torch.stack(validation_targets)
                              .flatten()
                              .detach()
                              .numpy())

        self.log_validation_data(targets=validation_targets)

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
                castling = (batch[self.castling_column]
                            .to(self.device))
                active_color = (batch[self.active_color_column]
                                .to(self.device))

                targets = self.gamma_discounted_targets(
                    total_moves=batch[self.total_moves_column],
                    winner=batch[self.winner_column],
                    active_color=batch[self.active_color_column],
                    game_len=batch[self.game_len_column]
                ).to(self.device)

                self.model.train()
                outputs = (self.model((boards, active_color, castling))
                           .flatten()
                           )

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
                    validation_outputs = self.forward_validation_data(val_dataloader=val_dataloader)

                    self.log_eval(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        outputs=validation_outputs,
                        targets=validation_targets,
                        len_trainset=len(train_dataloader),
                        plot_bins=list(np.arange(-1, 1.01, 0.1)),
                        hparams={
                            "mode": "distill_stockfish",
                            "model": self.model.__class__.__name__,
                            "model_hash": self.model.model_hash(),
                            "optimizer": self.optimizer.__class__.__name__,
                            "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
                            "gamma": self.gamma,
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

        self.writer.close()
