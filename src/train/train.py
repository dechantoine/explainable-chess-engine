from loguru import logger
from torch.nn import MSELoss
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from src.data.dataset import ChessBoardDataset
from src.models.simple_feed_forward import SimpleFF
from src.train.train_utils import init_training, train_test_split, training_loop

train_params = {
    "run_name": "simple_ff_17",
    "dataset_num_workers": 8,
    "dataloaders_num_workers": 2,
    "train_size": 0.90,
    "n_epochs": 20,
    "batch_size": 64,
    "lr": 0.1,
    "gamma": 0.99,
    "log_sampling": 0.05,
    "eval_sampling": 1,
}


def collate_fn(x):
    """Collate function for the DataLoader.

    Used to avoid using lambda function, thus allowing for
    multiprocessing.

    """
    return x


if __name__ == "__main__":
    logger.info("Initializing model, optimizer, and loss.")
    model = SimpleFF()
    optimizer = Adadelta(
        model.parameters(),
        lr=train_params["lr"],
    )
    loss = MSELoss()

    model, optimizer, resume_step = init_training(
        train_params["run_name"], model, optimizer
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
        num_workers=train_params["dataset_num_workers"],
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Splitting data.")
    train_dataset, test_dataset = train_test_split(
        dataset=dataset, seed=0, train_size=train_params["train_size"], stratify=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_params["dataloaders_num_workers"],
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_params["dataloaders_num_workers"],
    )

    logger.info("Starting training loop.")
    training_loop(
        model=model,
        optimizer=optimizer,
        loss=loss,
        gamma=train_params["gamma"],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        n_epochs=train_params["n_epochs"],
        resume_step=resume_step,
        device="cpu",
        log_sampling=train_params["log_sampling"],
        eval_sampling=train_params["eval_sampling"],
        run_name=train_params["run_name"],
    )
