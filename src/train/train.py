import click
from loguru import logger
from torch.nn import MSELoss
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from src.data.dataset import ChessBoardDataset
from src.models.simple_feed_forward import SimpleFF
from src.train.train_utils import init_training, train_test_split, training_loop


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
@click.argument("run_name")
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
       dataset_num_workers,
       dataloaders_num_workers,
       train_size,
       n_epochs,
       batch_size,
       lr,
       gamma,
       log_sampling,
       eval_sampling):

    logger.info("Initializing model, optimizer, and loss.")
    model = SimpleFF()
    optimizer = Adadelta(
        params=model.parameters(),
        lr=lr,
    )
    loss = MSELoss()

    model, optimizer, resume_step = init_training(
        run_name=run_name,
        model=model,
        optimizer=optimizer
    )

    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    logger.info("Loading data.")
    dataset = ChessBoardDataset(
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
    training_loop(
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
    )


if __name__ == "__main__":
    cli.add_command(rl)
    cli()
