import os
from cProfile import Profile
from pstats import SortKey, Stats

import click

from src.data.dataset import ChessBoardDataset


class ChessBoardProfiling:
    def __init__(self, data_dir, n_test=100):
        self.dataset = None
        self.n_test = n_test
        self.data_dir = data_dir

    def init(self):
        self.dataset = ChessBoardDataset(
            root_dir=self.data_dir,
            return_moves=False,
            return_outcome=False,
            transform=True,
            include_draws=False,
            in_memory=True,
            num_workers=8,
        )

    def get_boards_indices(self):
        for i in range(self.n_test):
            _ = self.dataset.get_boards_indices(include_draws=True)

    def retrieve_board(self):
        for i in range(self.n_test):
            _, _, _, _ = self.dataset.retrieve_board(0)

    def getitems(self):
        for i in range(self.n_test):
            _ = self.dataset.__getitems__(list(range(64)))


@click.group()
def cli():
    pass


@click.command()
@click.option("--save_dir", default="profile_package/profile_dataset", help="Directory to save profiling results.")
@click.option("--data_dir", default="profile_package/profile_data", help="Directory containing the dataset.")
@click.option("--n_test", default=100, help="Number of tests to run.")
def dataset(data_dir, save_dir, n_test):
    chess_profiling = ChessBoardProfiling(data_dir=data_dir, n_test=n_test)

    os.makedirs(save_dir, exist_ok=True)

    with Profile() as profiler:
        profiler.runcall(chess_profiling.init)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "init.prof"))

    with Profile() as profiler:
        profiler.runcall(chess_profiling.get_boards_indices)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "get_boards_indices.prof"))

    with Profile() as profiler:
        profiler.runcall(chess_profiling.retrieve_board)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "retrieve_board.prof"))

    with Profile() as profiler:
        profiler.runcall(chess_profiling.getitems)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "getitems.prof"))


if __name__ == "__main__":
    cli.add_command(dataset)
    cli()
