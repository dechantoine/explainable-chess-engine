import os
from cProfile import Profile
from pstats import SortKey, Stats

import click
import torch

from src.data.pgn_dataset import PGNDataset
from src.engine.agents.dl_agent import DLAgent
from src.engine.agents.stockfish_agent import StockfishAgent
from src.engine.games import Match


class ChessBoardProfiling:
    def __init__(self, data_dir, n_test=100):
        self.dataset = None
        self.n_test = n_test
        self.data_dir = data_dir

    def init(self):
        self.dataset = PGNDataset(
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


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=12 * 8 * 8, out_features=1)

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        return self.linear(x)


class MatchProfiling:
    def __init__(self, n_test=10, max_workers=8):
        stockfish_agent = StockfishAgent(is_white=True)
        dl_agent = DLAgent(model=MockModel(), is_white=False)
        self.n_test = n_test
        self.max_workers = max_workers
        self.match = Match(player_1=stockfish_agent, player_2=dl_agent, n_games=self.n_test)

    def play(self):
        self.match.play()

    def parallel_play(self):
        self.match.parallel_play(max_workers=self.max_workers)


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


@click.command()
@click.option("--save_dir", default="profile_package/profile_match", help="Directory to save profiling results.")
@click.option("--n_test", default=100, help="Number of tests to run.")
@click.option("--max_workers", default=8, help="Number of workers to use.")
def match(save_dir, n_test, max_workers):
    match_profiling = MatchProfiling(n_test=n_test, max_workers=max_workers)

    os.makedirs(save_dir, exist_ok=True)

    with Profile() as profiler:
        profiler.runcall(match_profiling.play)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "play.prof"))

    with Profile() as profiler:
        profiler.runcall(match_profiling.parallel_play)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(os.path.join(save_dir, "parallel_play.prof"))


if __name__ == "__main__":
    cli.add_command(dataset)
    cli.add_command(match)
    cli()
