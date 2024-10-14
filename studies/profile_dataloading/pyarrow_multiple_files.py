import os
import shutil
from timeit import timeit

import numpy as np
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

lines_per_file = [batch_size * 100000,
                  batch_size * 10000,
                  batch_size * 1000,
                  batch_size * 100]

results = pd.DataFrame(columns=["n_samples",
                                "lines_per_file",
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

    for n_lines in lines_per_file:
        dataset = ParquetChessDataset(path="./parquet_data",
                                      stockfish_eval=True,
                                      move_count=True)

        np.random.shuffle(dataset.indices)

        if os.path.exists("temp/train_dataset"):
            shutil.rmtree("temp/train_dataset")


        time_to_persist = timeit(lambda: dataset.persist(path="temp/train_dataset",
                                                         multiple_files=True,
                                                         lines_per_file=n_lines), number=1)

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
                                                     "lines_per_file": n_lines,
                                                     "time_to_persist": time_to_persist,
                                                     "n_workers": 1,
                                                     "prefetch_factor": 2,
                                                     "persistent_workers": True,
                                                     "batch_size": batch_size,
                                                     "time": persisted_time,
                                                     "batches_per_second": n_iter / persisted_time}])],
                            ignore_index=True)

    results.reset_index(drop=True, inplace=True)
    results.to_csv("studies/results/dataset_multiple_file.csv")
