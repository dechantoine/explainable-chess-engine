import os
from cProfile import Profile
from pstats import SortKey, Stats

from src.data.dataset import ChessBoardDataset


class ChessBoardProfiling:
    def __init__(self):
        self.dataset = None
        self.n_test = 100

    def init(self):
        self.dataset = ChessBoardDataset(
            root_dir="profile_package/profile_data",
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


if __name__ == "__main__":
    chess_profiling = ChessBoardProfiling()

    os.makedirs("./dataset", exist_ok=True)

    with Profile() as profiler:
        profiler.runcall(chess_profiling.init)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats("./dataset/init.prof")

    with Profile() as profiler:
        profiler.runcall(chess_profiling.get_boards_indices)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats("./dataset/get_boards_indices.prof")

    with Profile() as profiler:
        profiler.runcall(chess_profiling.retrieve_board)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats("./dataset/retrieve_board.prof")

    with Profile() as profiler:
        profiler.runcall(chess_profiling.getitems)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats("./dataset/getitems.prof")
