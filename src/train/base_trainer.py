import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from MultiChoice import MultiChoice
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CHECKPOINT_PREFIX = "checkpoint_"
START_FROM_SCRATCH = "Start From Scratch"


def online_mean(mean: float,
                new_values: np.array,
                current_size: int) -> float:
    """Compute the online mean.

    Args:
        mean (float): Current mean.
        new_values (np.array): New values.
        current_size (int): Current size.

    Returns:
        float: New mean.

    """
    return (mean * current_size + np.sum(new_values)) / (current_size + len(new_values))

class BaseTrainer(ABC):
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
                 clip_min: float = -10,
                 clip_max: float = 10
                 ):
        """Initializes the BaseTrainer class.

        Args:
            run_name (str): Name of the run.
            checkpoint_dir (str): Directory for checkpoints.
            log_dir (str): Directory for logs.
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            loss (torch.nn.Module): Loss function to use.
            device (torch.device): Device to use.
            log_sampling (float): Fraction of epoch to log.
            eval_sampling (float): Fraction of epoch to run and log eval.
            clip_min (float): Minimum value to clip.
            clip_max (float): Maximum value to clip.

        """
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.log_sampling = log_sampling
        self.eval_sampling = eval_sampling

        self.resume_step = 0
        self.log_interval = 0
        self.eval_interval = 0
        self.first_epoch = 0
        self.first_batch = 0
        self.running_loss = 0

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.plot_bins = list(np.arange(self.clip_min, self.clip_max + 0.1, 0.1))

        self.writer = None

        self.init_training()

    def online_bins(self,
                    predictions_bins_count: list[int],
                    targets_bins_count: list[int],
                    cross_bins_count: list[int],
                    predictions: np.array,
                    targets: np.array) -> tuple[list[int], list[int], list[int]]:
        """Compute the online bins.

        Args:
            predictions_bins_count (list[float]): Current count of predictions.
            targets_bins_count (list[float]): Current count of targets.
            cross_bins_count (list[float]): Current count of cross bins.
            predictions (np.array): Predictions.
            targets (np.array): Targets.

        Returns:
            list[float]: Online bins.

        """
        new_predictions_counts = list(np.histogram(predictions, bins=self.plot_bins)[0])
        new_targets_counts = list(np.histogram(targets, bins=self.plot_bins)[0])

        new_cross_counts = list(np.histogram2d(predictions, targets, bins=[self.plot_bins, self.plot_bins])[0].flatten())

        return (
            [pred + new_pred for pred, new_pred in zip(predictions_bins_count, new_predictions_counts)],
            [targ + new_targ for targ, new_targ in zip(targets_bins_count, new_targets_counts)],
            [cross + new_cross for cross, new_cross in zip(cross_bins_count, new_cross_counts)],
        )

    def plot_bivariate_distributions(self,
                                     cross_count: list[int]) -> plt.Figure:
        """Plot bivariate distributions of predictions and targets.

        Args:
            cross_count (list[int]): Cross count of predictions and targets.

        Returns:
            fig: plt.Figure, figure of the plot

        """
        sns.set_theme(style="darkgrid")

        axes = sns.histplot(
            data={
                "predictions": np.repeat(self.plot_bins[:-1], len(self.plot_bins) - 1),
                "targets": np.tile(self.plot_bins[:-1], len(self.plot_bins) - 1),
            },
            x="targets",
            y="predictions",
            weights=cross_count,
            stat="density",
            bins=self.plot_bins,
            cbar=True,
        )

        return axes.figure

    def list_existing_models(self) -> list[int]:
        """List the existing models for the given run name.

        Returns:
            list[int]: List of existing checkpoints.

        """
        list_checkpoints = []

        if os.path.exists(f"{self.checkpoint_dir}/{self.run_name}"):
            for f in os.listdir(f"{self.checkpoint_dir}/{self.run_name}"):
                if os.path.isfile(os.path.join(f"{self.checkpoint_dir}/{self.run_name}", f)):
                    list_checkpoints.append(int(f.split("_")[-1].split(".")[0]))

        list_checkpoints.sort()

        return list_checkpoints

    def ask_user_for_checkpoint(self) -> str or None:
        """Ask the user for the checkpoint to load.

        Returns:
            str: Checkpoint to load.

        """
        checkpoints = [str(chkpt) for chkpt in self.list_existing_models()]

        if len(checkpoints) == 0:
            return None

        else:
            question = MultiChoice(
                query="There exist the following checkpoints for this run name. Please choose one to load or start from "
                      "scratch:",
                options=[START_FROM_SCRATCH] + checkpoints,
            )
            answer = question()

        return answer

    def init_training(self) -> None:
        """Initialize the training."""
        checkpoint = self.ask_user_for_checkpoint()

        if checkpoint == "Start From Scratch":
            checkpoint = -1
            if os.path.exists(f"./{self.log_dir}/{self.run_name}"):
                logger.info(f"Removing existing further logs for {self.run_name}.")
                for f in os.listdir(f"./{self.log_dir}/{self.run_name}"):
                    os.remove(os.path.join(f"./{self.log_dir}/{self.run_name}", f))
                os.rmdir(f"./{self.log_dir}/{self.run_name}")

        elif checkpoint is None:
            logger.info("No existing checkpoints found.")
            checkpoint = 0

        else:
            checkpoint = int(checkpoint)
            logger.info(f"Loading checkpoint {checkpoint}.")
            chkpt = torch.load(f"./{self.checkpoint_dir}/{self.run_name}/{CHECKPOINT_PREFIX}{checkpoint}.pt")
            self.model.load_state_dict(chkpt["model_state_dict"])
            self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])

        if os.path.exists(f"./{self.checkpoint_dir}/{self.run_name}"):
            logger.info(f"Removing existing further checkpoints for {self.run_name}.")
            for f in os.listdir(f"./{self.checkpoint_dir}/{self.run_name}"):
                if int(f.split("_")[-1].split(".")[0]) > checkpoint:
                    os.remove(os.path.join(f"./{self.checkpoint_dir}/{self.run_name}", f))

        self.resume_step = max(checkpoint, 0)

    def init_loop_config(self,
                         len_train: int):
        """Initialize the loop configuration.

        Args:
            len_train (int): Length of the training set.

        """
        self.log_interval = int(len_train * self.log_sampling) + 1
        self.eval_interval = int(len_train * self.eval_sampling) + 1

        self.first_epoch = self.resume_step // len_train
        self.first_batch = self.resume_step % len_train

        self.writer = SummaryWriter(
            log_dir=f"./{self.log_dir}/{self.run_name}",
            purge_step=self.first_epoch * len_train + self.first_batch + 1,
        )

    def validation(self,
                   outputs: np.array = None,
                   targets: np.array = None,
                   dict_metrics: dict[str, float] = None,
                   ) -> dict[str, float]:
        """Validation metrics for the model.

        Args:
            outputs (np.array): Outputs of the model.
            targets (np.array): Targets of the model.
            dict_metrics (dict[str, float]): Evaluation metrics computed online.

        Returns:
            dict[str, float]: Evaluation metrics

        """
        if dict_metrics:
            eval_metrics = {}
            eval_metrics["Distributions/mean_pred_%"] = dict_metrics["mean_outputs"] / dict_metrics["mean_targets"]

            eval_metrics["Distributions/std_pred_%"] = (
                    ((dict_metrics["mean_squared_outputs"] - dict_metrics["mean_outputs"] ** 2) ** 0.5) /
                    ((dict_metrics["mean_squared_targets"] - dict_metrics["mean_targets"] ** 2) ** 0.5)
            )

            eval_metrics["Errors/mean_error"] = dict_metrics["mean_errors"]

            eval_metrics["Errors/std_error"] = (
                    (dict_metrics["mean_squared_errors"] - dict_metrics["mean_errors"] ** 2) ** 0.5
            )

            eval_metrics["Errors/root_mean_squared_error"] = dict_metrics["mean_squared_errors"] ** 0.5

            eval_metrics["Errors/mean_absolute_error"] = dict_metrics["mean_absolute_errors"]

            eval_metrics["Errors/std_absolute_error"] = (
                    (dict_metrics["mean_squared_errors"] - dict_metrics["mean_absolute_errors"] ** 2) ** 0.5
            )

            return eval_metrics

        eval_metrics = {}
        errors = (targets - outputs).flatten()
        squared_errors = errors ** 2

        eval_metrics["Distributions/mean_pred_%"] = np.mean(outputs) / np.mean(
            targets
        )
        eval_metrics["Distributions/std_pred_%"] = np.std(outputs) / np.std(targets)

        eval_metrics["Errors/mean_error"] = np.mean(errors)
        eval_metrics["Errors/std_error"] = np.std(errors)
        eval_metrics["Errors/root_mean_squared_error"] = np.sqrt(np.mean(squared_errors))
        eval_metrics["Errors/mean_absolute_error"] = np.mean(abs(errors))
        eval_metrics["Errors/std_absolute_error"] = np.std(abs(errors))

        return eval_metrics

    def online_validation(self,
                          batch_outputs: np.array,
                          batch_targets: np.array,
                          dict_metrics: dict[str, float] = None,
                          ) -> dict[str, float]:
        """Validation metrics for the model computed online.

        Args:
            batch_outputs (np.array): Outputs of the model.
            batch_targets (np.array): Targets of the model.
            dict_metrics (dict[str, float]): Current evaluation metrics.

        Returns:
            dict[str, dict[str, float]]: Evaluation metrics

        """
        if not dict_metrics:
            dict_metrics = {
                "n_samples": len(batch_outputs),
                "mean_outputs": np.mean(batch_outputs),
                "mean_targets": np.mean(batch_targets),
                "mean_squared_outputs": np.mean(batch_outputs ** 2),
                "mean_squared_targets": np.mean(batch_targets ** 2),
                "mean_errors": np.mean(batch_targets - batch_outputs),
                "mean_squared_errors": np.mean((batch_targets - batch_outputs) ** 2),
                "mean_absolute_errors": np.mean(abs(batch_targets - batch_outputs)),
            }

            return dict_metrics

        errors = (batch_targets - batch_outputs).flatten()

        dict_metrics["mean_outputs"] = online_mean(
            mean=dict_metrics["mean_outputs"],
            new_values=batch_outputs,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_targets"] = online_mean(
            mean=dict_metrics["mean_targets"],
            new_values=batch_targets,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_squared_outputs"] = online_mean(
            mean=dict_metrics["mean_squared_outputs"],
            new_values=batch_outputs ** 2,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_squared_targets"] = online_mean(
            mean=dict_metrics["mean_squared_targets"],
            new_values=batch_targets ** 2,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_errors"] = online_mean(
            mean=dict_metrics["mean_errors"],
            new_values=errors,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_squared_errors"] = online_mean(
            mean=dict_metrics["mean_squared_errors"],
            new_values=errors ** 2,
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["mean_absolute_errors"] = online_mean(
            mean=dict_metrics["mean_absolute_errors"],
            new_values=abs(errors),
            current_size=dict_metrics["n_samples"],
        )

        dict_metrics["n_samples"] += len(batch_outputs)

        return dict_metrics

    def training_step(self,
                      outputs: torch.Tensor,
                      targets: torch.Tensor
                      ) -> float:
        """Training step for the model.

        Args:
            outputs (torch.Tensor): Outputs of the model.
            targets (torch.Tensor): Targets of the model.

        Returns:
            float: Loss value.

        """
        self.optimizer.zero_grad()
        loss_value = self.loss(outputs, targets)
        loss_value.backward()
        self.optimizer.step()

        return loss_value.item()

    def log_train(self,
                  epoch: int,
                  batch_idx: int,
                  len_trainset: int,
                  ) -> None:
        """Log the training step.

        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch.
            len_trainset (int): Length of the training set.

        """
        if batch_idx == 0:
            running_loss = self.running_loss
        else:
            running_loss = self.running_loss / self.log_interval
        self.writer.add_scalar(
            tag="Training/loss",
            scalar_value=running_loss,
            global_step=epoch * len_trainset + batch_idx,
        )

    def log_eval(self,
                 epoch: int,
                 batch_idx: int,
                 val_dataloader: torch.utils.data.DataLoader,
                 len_trainset: int,
                 hparams: dict,
                 ) -> None:
        """Log the evaluation step.

        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch.
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
            len_trainset (int): Length of the training set.
            hparams (dict): Hyperparameters to log.

        """
        if batch_idx == len_trainset - 1:
            logger.info(f"Running eval on the end of epoch {epoch}...")
            global_step = (epoch + 1) * len_trainset

        elif batch_idx == 0 and epoch > 0:
            return

        else:
            logger.info(f"Running eval on epoch {epoch}, batch {batch_idx}...")
            global_step = epoch * len_trainset + batch_idx

        dict_metrics = None
        bins_count_outputs = [0] * (len(self.plot_bins)-1)
        bins_count_targets = [0] * (len(self.plot_bins)-1)
        bins_count_cross = [0] * ((len(self.plot_bins)-1) ** 2)

        for batch in tqdm(iterable=val_dataloader,
                          desc="Validation batches",):

            outputs = self.forward_validation_data(batch)
            targets = (batch["stockfish_eval"]
                       .flatten()
                       .clip(min=self.clip_min, max=self.clip_max)
                       .detach()
                       .numpy()
                       )

            dict_metrics = self.online_validation(
                batch_outputs=outputs,
                batch_targets=targets,
                dict_metrics=dict_metrics,
            )

            bins_count_outputs, bins_count_targets, bins_count_cross = self.online_bins(
                predictions_bins_count=bins_count_outputs,
                targets_bins_count=bins_count_targets,
                cross_bins_count=bins_count_cross,
                predictions=outputs,
                targets=targets,
            )

        val_metrics = self.validation(
            dict_metrics=dict_metrics,
        )

        self.writer.add_histogram_raw(
            tag="Distributions/outputs",
            min=self.clip_min,
            max=self.clip_max,
            num=sum(bins_count_outputs),
            sum=dict_metrics["mean_outputs"] * dict_metrics["n_samples"],
            sum_squares=dict_metrics["mean_squared_outputs"] * dict_metrics["n_samples"],
            bucket_limits=self.plot_bins[1:],
            bucket_counts=bins_count_outputs,
            global_step=global_step,
        )

        fig = self.plot_bivariate_distributions(cross_count=bins_count_cross)
        self.writer.add_figure(
            tag="Distributions/bivariate", figure=fig, global_step=global_step
        )

        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict=val_metrics,
            run_name=".",
            global_step=global_step,
        )

        self.writer.close()

        if not os.path.exists(f"./{self.checkpoint_dir}/{self.run_name}"):
            os.makedirs(f"./{self.checkpoint_dir}/{self.run_name}")

        torch.save(
            obj={
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            f=f"./{self.checkpoint_dir}/{self.run_name}/{CHECKPOINT_PREFIX}{global_step}.pt",
        )

    def log_validation_data(self,
                            targets: np.array
                            ) -> None:
        """Log the test data.

        Args:
            targets (np.array): Targets to log.

        """
        logger.info("Logging validation data...")

        self.writer.add_histogram(
            tag="ValidationData/targets_distribution", bins="auto", values=targets, global_step=0
        )

    def forward_validation_data(self, batch) -> np.array:
        """Forward the validation data.

        Args:
            batch: Validation batch.

        Returns:
            np.array: Validation outputs.

        """
        self.model.eval()
        pass

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
        pass
