import click
from loguru import logger
from torch.nn import MSELoss
from torch.optim import Adadelta, AdamW, lr_scheduler
from torch.utils.data import DataLoader

from src.data.parquet_dataset import ParquetChessDataset
from src.models import get_model
from src.models.multi_input_attention import AttentionModel
from src.train.distill import ChessEvalLoss, DistillTrainer
from src.train.reinforcement import RLTrainer


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

    trainer = RLTrainer(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model=model,
        optimizer=optimizer,
        loss=loss,
        device="cpu",
        log_sampling=log_sampling,
        eval_sampling=eval_sampling,
        board_column="board",
        active_color_column="active_color",
        castling_column="castling",
        winner_column="winner",
        total_moves_column="total_moves",
        game_len_column="game_len",
        gamma=gamma,
    )

    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    logger.info("Loading data.")
    dataset = ParquetChessDataset(path="./parquet_data",
                                  stockfish_eval=False)
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Splitting data.")
    train_dataset, val_dataset = trainer.train_test_split(
        dataset=dataset,
        seed=0,
        train_size=train_size,
        stratify=True
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
    trainer.training_loop(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        n_epochs=n_epochs,
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

    #model = get_model(model_name)
    model = AttentionModel()

    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        #amsgrad=True,
    )

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                        T_max=5,
                                        eta_min=0.000001,
                                        verbose=True)

    loss = ChessEvalLoss()

    trainer = DistillTrainer(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        device="cpu",
        log_sampling=log_sampling,
        eval_sampling=eval_sampling,
        eval_column="stockfish_eval",
        board_column="board",
        active_color_column="active_color",
        castling_column="castling",
        clip_min=-50.0,
        clip_max=50.0,
    )

    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    logger.info("Loading data.")
    dataset = ParquetChessDataset(path="./parquet_data",
                                  stockfish_eval=True,
                                  move_count=True)

    #dataset.downsampling(seed=42, ratio=0.01)

    logger.info(f"Dataset size: {len(dataset)}")

    dataset = trainer.balanced_eval_signs(dataset=dataset)

    logger.info("Splitting data.")
    train_dataset, val_dataset = trainer.train_test_split(
        dataset=dataset,
        seed=0,
        train_size=train_size,
        stratify=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
        prefetch_factor=8,
        #multiprocessing_context="forkserver",
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dataloaders_num_workers,
        prefetch_factor=8,
        #multiprocessing_context="forkserver",
        persistent_workers=False
    )

    logger.info("Starting training loop.")
    trainer.training_loop(train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          n_epochs=n_epochs)


if __name__ == "__main__":
    cli.add_command(rl)
    cli.add_command(distill)
    cli()
