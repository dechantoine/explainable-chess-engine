from loguru import logger
from torch.nn import MSELoss
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from src.data.dataset import ChessBoardDataset
from src.models.simple_feed_forward import SimpleFF
from src.train.train_utils import train_test_split, training_loop

if __name__ == "__main__":
    logger.info("Loading data.")
    dataset = ChessBoardDataset(
        root_dir="./sample_data",
        transform=True,
        return_moves=True,
        return_outcome=True,
        include_draws=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Splitting data.")
    train_dataset, test_dataset = train_test_split(
        dataset=dataset, seed=0, train_size=0.90, stratify=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=lambda x: x
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=True, collate_fn=lambda x: x
    )

    logger.info("Initializing model, optimizer, and loss.")
    model = SimpleFF()
    optimizer = Adagrad(
        model.parameters(),
        lr=0.05,
        lr_decay=0.001,
    )
    loss = MSELoss()
    logger.info(
        f"Model number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # with profile(activities=[ProfilerActivity.CPU],
    #             record_shapes=True) as prof:

    logger.info("Starting training loop.")
    training_loop(
        model=model,
        optimizer=optimizer,
        loss=loss,
        gamma=0.99,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        n_epochs=10,
        device="cpu",
        log_sampling=0.05,
        eval_sampling=0.25,
        run_name="simple_ff_O",
    )

    # logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
