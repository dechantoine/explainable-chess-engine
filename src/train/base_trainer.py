import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from MultiChoice import MultiChoice
from torch.utils.tensorboard import SummaryWriter

CHECKPOINT_PREFIX = "checkpoint_"
START_FROM_SCRATCH = "Start From Scratch"


def plot_bivariate_distributions(predictions: np.array,
                                 targets: np.array,
                                 bins: list[float]
                                 ) -> plt.Figure:
    """Plot bivariate distributions of predictions and targets.

    Args:
        predictions (np.array): predictions.
        targets (np.array): targets.
        bins (list[float]): bins for the plot.

    Returns:
        fig: plt.Figure, figure of the plot

    """
    sns.set_theme(style="darkgrid")

    axes = sns.histplot(
        data={"targets": targets, "predictions": predictions},
        x="targets",
        y="predictions",
        stat="density",
        bins=bins,
        cbar=True,
    )

    return axes.figure


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
                 eval_sampling: float):
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

        self.writer = None

        self.init_training()

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
        self.log_interval = int(len_train * self.log_sampling)
        self.eval_interval = int(len_train * self.eval_sampling)

        self.first_epoch = self.resume_step // len_train
        self.first_batch = self.resume_step % len_train

        self.writer = SummaryWriter(
            log_dir=f"./{self.log_dir}/{self.run_name}",
            purge_step=self.first_epoch * len_train + self.first_batch + 1,
        )

    def validation(self,
                   outputs: np.array,
                   targets: np.array,
    ) -> dict[str, dict[str, float]]:
        """Validation metrics for the model.

        Args:
            outputs (np.array): Outputs of the model.
            targets (np.array): Targets of the model.

        Returns:
            dict[str, dict[str, float]]: Evaluation metrics

        """
        eval_scalars = {}
        errors = (targets - outputs).flatten()
        squared_errors = errors ** 2

        eval_scalars["Distributions/mean_pred_%"] = np.mean(outputs) / np.mean(
            targets
        )
        eval_scalars["Distributions/std_pred_%"] = np.std(outputs) / np.std(targets)

        eval_scalars["Errors/mean_error"] = np.mean(errors)
        eval_scalars["Errors/std_error"] = np.std(errors)
        eval_scalars["Errors/root_mean_squared_error"] = np.sqrt(np.mean(squared_errors))
        eval_scalars["Errors/root_std_squared_error"] = np.sqrt(np.std(squared_errors))
        eval_scalars["Errors/mean_absolute_error"] = np.mean(abs(errors))
        eval_scalars["Errors/std_absolute_error"] = np.std(abs(errors))

        return eval_scalars

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
                 outputs: np.array,
                 targets: np.array,
                 len_trainset: int,
                 plot_bins: list,
                 hparams: dict,
                 ) -> None:
        """Log the evaluation step.

        Args:
            epoch (int): Current epoch.
            batch_idx (int): Current batch.
            outputs (np.array): Validation outputs of the model.
            targets (np.array): Validation targets of the model.
            len_trainset (int): Length of the training set.
            plot_bins (list): Bins for the plot.
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

        val_metrics = self.validation(
            outputs=outputs, targets=targets,
        )

        self.writer.add_histogram(
            tag="Distributions/outputs",
            bins="auto",
            values=outputs,
            global_step=global_step,
        )

        fig = plot_bivariate_distributions(predictions=outputs, targets=targets, bins=plot_bins)
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

    def forward_validation_data(self, val_dataloader: torch.utils.data.DataLoader) -> np.array:
        """Forward the validation data.

        Args:
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader.

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
