import torch

from src.data.dataset import ChessBoardDataset
from src.models.simple_feed_forward import SimpleFF
from src.train.train_utils import train_test_split, reward_fn

from loguru import logger

if __name__ == "__main__":
    dataset = ChessBoardDataset(root_dir='./sample_data',
                                transform=True,
                                return_moves=True,
                                return_outcome=True,
                                include_draws=False)

    train_dataset, test_dataset = train_test_split(dataset=dataset,
                                                   seed=0,
                                                   train_size=0.8)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   collate_fn=lambda x: x)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  collate_fn=lambda x: x)

    model = SimpleFF()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    for batch in train_dataloader:
        boards, moves, outcomes = batch
        optimizer.zero_grad()
        pred = model(boards).reshape(-1)
        targets = reward_fn(outcome=outcomes, gamma=0.99)
        loss_value = loss(pred, targets)
        loss_value.backward()
        optimizer.step()
        logger.info(f'Loss: {loss_value.item()}')

