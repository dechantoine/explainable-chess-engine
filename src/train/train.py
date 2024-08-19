import click
from loguru import logger
from torch.nn import MSELoss
from torch.optim import Adadelta, Adam
from torch.utils.data import DataLoader

from src.data.parquet_dataset import ParquetChessDataset
from src.data.pgn_dataset import PGNDataset
from src.models import get_model
from src.train.train_utils import (
    ChessEvalLoss,
    init_training,
    train_test_split,
    training_loop_distill,
    training_loop_rl,
)


def collate_fn(x):
    """Collate function for the DataLoader.

    Used to avoid using lambda function, thus allowing for
    multiprocessing.

    """
    return x


@click.group()
def cli():
    pass


@click.command()
@click.option("--run_name", required=True, help="Run name.")
@click.option("--model_name", required=True, help="Model name.")
@click.option("--checkpoint_dir", default="models_checkpoint", help="Checkpoint directory.")
@click.option("--log_dir", default="logs", help="Log directory.")
@click.option("--dataset_num_workers", default=8, help="Python dataset number of workers.")
@click.option("--dataloaders_num_workers", default=2, help="Torch Dataloaders number of workers.")
@click.option("--train_size", default=0.9, help="Train size.")
@click.option("--n_epochs", default=20, help="Number of epochs.")
@click.option("--batch_size", default=64, help="Batch size.")
@click.option("--lr", default=0.1, help="Learning rate.")
@click.option("--gamma", default=0.99, help="Gamma discount factor.")
@click.option("--log_sampling", default=0.05, help="Log every x fraction of epoch.")
@click.option("--eval_sampling", default=1.0, help="Run and log eval every x fraction of epoch.")
def rl(run_name,
       model_name,
       checkpoint_dir,
       log_dir,
       dataset_num_workers,
       dataloaders_num_workers,
       train_size,
       n_epochs,
       batch_size,
       lr,
       gamma,
       log_sampling,
       eval_sampling):
    logger.info("Initializing RL: model, optimizer, and loss.")
    model = get_model(model_name)
    optimizer = Adadelta(
        params=model.parameters(),
        lr=lr,
    )
    loss = MSELoss()

    model, optimizer, resume_step = init_training(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model=model,
        optimizer=optimizer
    )

    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    logger.info("Loading data.")
    dataset = PGNDataset(
        root_dir="./sample_data",
        transform=True,
        return_moves=False,
        return_outcome=True,
        include_draws=False,
        in_memory=True,
        num_workers=dataset_num_workers,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Splitting data.")
    train_dataset, test_dataset = train_test_split(
        dataset=dataset,
        seed=0,
        train_size=train_size,
        stratify=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
    )

    logger.info("Starting training loop.")
    training_loop_rl(
        model=model,
        optimizer=optimizer,
        loss=loss,
        gamma=gamma,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        n_epochs=n_epochs,
        resume_step=resume_step,
        device="cpu",
        log_sampling=log_sampling,
        eval_sampling=eval_sampling,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )


@click.command()
@click.option("--run_name", required=True, help="Run name.")
@click.option("--model_name", default="multi_input_feedforward", required=True, help="Model name.")
@click.option("--checkpoint_dir", default="models_checkpoints_distill", help="Checkpoint directory.")
@click.option("--log_dir", default="logs_distill", help="Log directory.")
@click.option("--dataloaders_num_workers", default=2, help="Torch Dataloaders number of workers.")
@click.option("--train_size", default=0.9, help="Train size.")
@click.option("--n_epochs", default=20, help="Number of epochs.")
@click.option("--batch_size", default=64, help="Batch size.")
@click.option("--lr", default=0.001, help="Learning rate.")
@click.option("--log_sampling", default=0.1, help="Log every x fraction of epoch.")
@click.option("--eval_sampling", default=1.0, help="Run and log eval every x fraction of epoch.")
def distill(run_name,
            model_name,
            checkpoint_dir,
            log_dir,
            dataloaders_num_workers,
            train_size,
            n_epochs,
            batch_size,
            lr,
            log_sampling,
            eval_sampling):
    logger.info("Initializing Distillation: model, optimizer, and loss.")
    model = get_model(model_name)
    optimizer = Adam(
        params=model.parameters(),
        lr=lr,
    )
    loss = ChessEvalLoss()

    model, optimizer, resume_step = init_training(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model=model,
        optimizer=optimizer
    )

    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    logger.info("Loading data.")
    dataset = ParquetChessDataset(path="./parquet_data",
                                  stockfish_eval=True)
    logger.info(f"Dataset size: {len(dataset)}")

    dataset.balanced_eval_signs()

    logger.info("Splitting data.")
    train_dataset, val_dataset = dataset.train_test_split(
        seed=0,
        train_size=train_size,
        stratify="stockfish_eval"
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
    )

    logger.info("Starting training loop.")
    training_loop_distill(
        model=model,
        optimizer=optimizer,
        loss=loss,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        n_epochs=n_epochs,
        resume_step=resume_step,
        device="cpu",
        log_sampling=log_sampling,
        eval_sampling=eval_sampling,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    cli.add_command(rl)
    cli.add_command(distill)
    cli()
