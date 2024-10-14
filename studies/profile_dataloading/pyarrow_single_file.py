from timeit import timeit

import pandas as pd
from loguru import logger
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.parquet_dataset import ParquetChessDataset
from src.models import get_model
from src.train.distill import DistillTrainer

n_iter = 100
batch_size = 64

downsampling = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

results = pd.DataFrame(columns=["single_file",
                                "n_samples",
                                "downsampling",
                                "time_to_persist",
                                "n_workers",
                                "prefetch_factor",
                                "persistent_workers",
                                "batch_size",
                                "time",
                                "batches_per_second"])


def collate_fn(x):
    return x

def iterate_dataloader(dataloader, n_iter):
    for i, batch in tqdm(iterable=enumerate(dataloader),
                         total=n_iter,
                         desc="Profiling dataloader",
                         unit="batch"):
        if i == n_iter:
            return


if __name__ == "__main__":
    logger.info("Initializing trainer.")

    model = get_model("multi_input_conv")

    trainer = DistillTrainer(
        run_name="distill",
        checkpoint_dir="models_checkpoint",
        log_dir="logs",
        model=model,
        optimizer=AdamW(
            params=model.parameters(),
            lr=0.1,
        ),
        loss=MSELoss(),
        device="cpu",
        log_sampling=0.05,
        eval_sampling=1.0,
        eval_column="stockfish_eval",
        board_column="board",
        active_color_column="active_color",
        castling_column="castling",
        clip_min=-50.0,
        clip_max=50.0,
    )

    for d in downsampling:
        dataset = ParquetChessDataset(path="./parquet_data",
                                      stockfish_eval=True,
                                      move_count=True)

        dataset.indices = dataset.indices[:int(batch_size * n_iter * 10000)]
        dataset.indices = dataset.indices[:int(len(dataset) * d)]

        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            prefetch_factor=2,
            # multiprocessing_context="forkserver",
            persistent_workers=True
        )

        logger.info("Profiling dataloaders.")

        unbalanced_time = timeit(lambda: iterate_dataloader(train_dataloader, n_iter), number=1)
        logger.info(f"Multi file dataset time: {unbalanced_time}")

        results = pd.concat([results, pd.DataFrame([{"single_file": False,
                                                     "n_samples": len(dataset),
                                                     "downsampling": d,
                                                     "time_to_persist": None,
                                                     "n_workers": 1,
                                                     "prefetch_factor": 2,
                                                     "persistent_workers": True,
                                                     "batch_size": batch_size,
                                                     "time": unbalanced_time,
                                                     "batches_per_second": n_iter / unbalanced_time}])],
                            ignore_index=True)

    for d in downsampling:
        dataset = ParquetChessDataset(path="./parquet_data",
                                      stockfish_eval=True,
                                      move_count=True)

        dataset.indices = dataset.indices[:(batch_size * n_iter) * 10000]

        time_to_persist = timeit(lambda: dataset.persist(path="temp/train_dataset.parquet"), number=1)

        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            prefetch_factor=2,
            # multiprocessing_context="forkserver",
            persistent_workers=True
        )

        persisted_time = timeit(lambda: iterate_dataloader(train_dataloader, n_iter), number=1)
        logger.info(f"Persisted dataset time: {persisted_time}")

        results = pd.concat([results, pd.DataFrame([{"single_file": True,
                                                     "n_samples": len(dataset),
                                                     "downsampling": d,
                                                     "time_to_persist": time_to_persist,
                                                     "n_workers": 1,
                                                     "prefetch_factor": 2,
                                                     "persistent_workers": True,
                                                     "batch_size": batch_size,
                                                     "time": persisted_time,
                                                     "batches_per_second": n_iter / persisted_time}])],
                            ignore_index=True)

    results.reset_index(drop=True, inplace=True)
    results.to_csv("studies/results/dataset_single_file.csv")
