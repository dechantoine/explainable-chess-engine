from src.data.dataset import ChessBoardDataset

from loguru import logger
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from tqdm import tqdm


@logger.catch
def train_test_split(dataset: ChessBoardDataset,
                     seed: int,
                     train_size: float) \
        -> (ChessBoardDataset, ChessBoardDataset):
    """Split the provided dataset into a training and testing set.

    Args:
        dataset (ChessBoardDataset): Dataset to split.
        seed (int): Seed for the random split.
        train_size (float): Proportion of the training set.

    Returns:
        ChessBoardDataset: Training dataset.
        ChessBoardDataset: Testing dataset.
    """
    np.random.seed(seed)

    indices = np.random.permutation(len(dataset))
    split = int(train_size * len(dataset))
    train_indices, test_indices = indices[:split], indices[split:]

    train_set = deepcopy(dataset)
    test_set = deepcopy(dataset)

    train_set.board_indices = [dataset.board_indices[i] for i in train_indices]
    test_set.board_indices = [dataset.board_indices[i] for i in test_indices]

    return train_set, test_set


@logger.catch
def reward_fn(outcome: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Calculate the reward for the given outcome.

    Args:
        outcome (torch.Tensor): Outcome of the game.
        gamma (float): Discount factor.

    Returns:
        tensor: Rewards for the given outcomes.
    """
    return torch.Tensor((gamma ** (outcome[:, 1] - outcome[:, 0])) * outcome[:, 2])


@logger.catch
def training_step(model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  loss: torch.nn.modules.loss._Loss,
                  boards: torch.Tensor,
                  outcomes: torch.Tensor,
                  gamma: float = 0.99,
                  return_pred: bool = False) -> float or tuple[float, torch.Tensor]:
    """Training step for the model.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer to use.
        loss (torch.nn.modules.loss._Loss): Loss function to use.
        boards (torch.Tensor): Input boards.
        outcomes (torch.Tensor): Outcomes of the games.
        gamma (float): Discount factor.
        return_pred (bool): Return the predictions.
    """
    optimizer.zero_grad()
    pred = model(boards).reshape(-1)
    targets = reward_fn(outcome=outcomes, gamma=gamma)
    loss_value = loss(pred, targets)
    loss_value.backward()
    optimizer.step()

    if return_pred:
        return loss_value.item(), pred

    return loss_value.item()


@logger.catch
def training_loop(model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  loss: torch.nn.modules.loss._Loss,
                  gamma: float,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader = None,
                  n_epochs: int = 1,
                  device: torch.device = torch.device("cpu"),
                  log_sampling: float = 0.1,
                  eval_sampling: float = 1,
                  run_name: str = "default"
                  ) -> None:
    """Training loop for the model.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer to use.
        loss (torch.nn.modules.loss._Loss): Loss function to use.
        gamma (float): Discount factor.
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        n_epochs (int): Number of epochs.
        device (torch.device): Device to use.
        test_dataloader (torch.utils.data.DataLoader): Testing dataloader.
        log_sampling (float): Sampling rate for logging.
        eval_sampling (float): Sampling rate for evaluation.
        run_name (str): Name of the run.
    """
    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    log_interval = int(len(train_dataloader) * log_sampling)
    eval_interval = int(len(train_dataloader) * eval_sampling)

    log_testdata(test_dataloader=test_dataloader,
                 gamma=gamma,
                 writer=writer)

    for epoch in tqdm(iterable=range(n_epochs),
                      desc="Epochs", ):
        running_loss = 0.0

        for batch_idx, batch in tqdm(iterable=enumerate(train_dataloader),
                                     desc="Batches",
                                     total=len(train_dataloader)):

            boards, moves, outcomes = batch
            boards = boards.to(device)
            outcomes = outcomes.to(device)

            loss_value = training_step(model=model,
                                       optimizer=optimizer,
                                       loss=loss,
                                       boards=boards,
                                       outcomes=outcomes)

            running_loss += loss_value

            if batch_idx % log_interval == 0:
                log_train(epoch=epoch,
                          batch_idx=batch_idx,
                          running_loss=running_loss,
                          len_trainset=len(train_dataloader),
                          log_interval=log_interval,
                          writer=writer)
                running_loss = 0.0

            if batch_idx % eval_interval == 0 and test_dataloader is not None:
                log_eval(epoch=epoch,
                         batch_idx=batch_idx,
                         len_trainset=len(train_dataloader),
                         model=model,
                         test_dataloader=test_dataloader,
                         gamma=gamma,
                         device=device,
                         writer=writer)

        if test_dataloader is not None:
            log_eval(epoch=epoch,
                     batch_idx=len(train_dataloader),
                     len_trainset=len(train_dataloader),
                     model=model,
                     test_dataloader=test_dataloader,
                     gamma=gamma,
                     device=device,
                     writer=writer)

    writer.close()


@logger.catch
def log_train(epoch: int,
              batch_idx: int,
              running_loss: float,
              len_trainset: int,
              log_interval: int,
              writer: SummaryWriter) -> None:
    """Log the training step.

    Args:
        epoch (int): Current epoch.
        batch_idx (int): Current batch.
        running_loss (float): Running loss.
        len_trainset (int): Length of the training set.
        log_interval (int): Interval for logging.
        writer (SummaryWriter): Writer for the logs.
    """
    if batch_idx == 0:
        running_loss = running_loss
    else:
        running_loss = running_loss / log_interval
    writer.add_scalar(tag="Training/loss",
                      scalar_value=running_loss,
                      global_step=epoch * len_trainset + batch_idx)
    logger.info(f"Epoch {epoch}, batch {batch_idx}, loss: {running_loss}")


@logger.catch
def log_eval(epoch: int,
             batch_idx: int,
             len_trainset: int,
             model: torch.nn.Module,
             test_dataloader: torch.utils.data.DataLoader,
             gamma: float,
             device: torch.device,
             writer: SummaryWriter) -> None:
    """Log the evaluation step.

    Args:
        epoch (int): Current epoch.
        batch_idx (int): Current batch.
        len_trainset (int): Length of the training set.
        model (torch.nn.Module): Model to evaluate.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the test set.
        gamma (float): Discount factor.
        device (torch.device): Device to use.
        writer (SummaryWriter): Writer for the logs.
    """
    if batch_idx == len_trainset:
        logger.info(f"Running eval on the end of epoch {epoch}...")
        global_step = (epoch + 1) * len_trainset

    elif batch_idx == 0 and epoch > 0:
        return

    else:
        logger.info(f"Running eval on epoch {epoch}, batch {batch_idx}...")
        global_step = epoch * len_trainset + batch_idx

    val_metrics, outputs = validation(model=model,
                                      test_dataloader=test_dataloader,
                                      gamma=gamma, device=device)

    for root_tag, metrics in val_metrics.items():
        for key, value in metrics.items():
            writer.add_scalar(tag=f"{root_tag}/{key}",
                              scalar_value=value,
                              global_step=global_step)

    writer.add_histogram(tag="Distributions/outputs",
                         bins="auto",
                         values=outputs,
                         global_step=global_step)


@logger.catch
def validation(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               gamma: float,
               device: torch.device) -> dict[str, dict[str, float]] and np.array:
    """Validation function for the model.

    Args:
        model (torch.nn.Module): Model to validate.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the validation set.
        gamma (float): Discount factor.
        device (torch.device): Device to use.

    Returns:
        dict[str, dict[str, float]], np.array: Evaluation metrics and outputs.
    """
    model.eval()
    val_targets = []
    val_outputs = []

    with torch.no_grad():
        for i, batch in tqdm(iterable=enumerate(test_dataloader, 0),
                             desc="Validation batches",
                             total=len(test_dataloader)):
            boards, moves, outcomes = batch
            boards = boards.to(device)

            targets = reward_fn(outcome=outcomes, gamma=gamma)

            outputs_nn = model(boards)
            val_targets.extend(targets.cpu().detach().numpy())
            val_outputs.extend(outputs_nn.cpu().detach().numpy())

    val_outputs = np.array(val_outputs).flatten()
    val_targets = np.array(val_targets).flatten()

    eval_scalars = {"Errors": {},
                    "Distributions": {}
                    }
    errors = (val_targets - val_outputs).flatten()
    squared_errors = errors ** 2

    eval_scalars["Distributions"]["mean_pred_%"] = np.mean(val_outputs) / np.mean(val_targets)
    eval_scalars["Distributions"]["std_pred_%"] = np.std(val_outputs) / np.std(val_targets)

    eval_scalars["Errors"]["mean_error"] = np.mean(errors)
    eval_scalars["Errors"]["std_error"] = np.std(errors)
    eval_scalars["Errors"]["root_mean_squared_error"] = np.sqrt(np.mean(squared_errors))
    eval_scalars["Errors"]["root_std_squared_error"] = np.sqrt(np.std(squared_errors))
    eval_scalars["Errors"]["mean_absolute_error"] = np.mean(abs(errors))
    eval_scalars["Errors"]["std_absolute_error"] = np.std(abs(errors))

    return eval_scalars, val_outputs


@logger.catch
def log_testdata(test_dataloader: torch.utils.data.DataLoader,
                 gamma: float,
                 writer: SummaryWriter) -> None:
    """Log the test data.

    Args:
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the test set.
        gamma (float): Discount factor.
        writer (SummaryWriter): Writer for the logs.
    """
    logger.info(f"Logging test data...")

    outcomes = []
    for i, batch in enumerate(test_dataloader):
        _, _, outcome = batch
        outcomes.extend(outcome)

    outcomes = torch.stack(outcomes)
    rewards = reward_fn(outcomes, gamma=gamma)

    writer.add_histogram(tag="TestData/rewards_distribution",
                         bins="auto",
                         values=rewards,
                         global_step=0)
