import os
from cProfile import Profile
from pstats import SortKey, Stats

from src.data.dataset import ChessBoardDataset

save_dir = "profile_package/profile_dataset"


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
