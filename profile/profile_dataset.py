from src.data.dataset import ChessBoardDataset

import os
from loguru import logger
from cProfile import Profile
from pstats import SortKey, Stats


class ChessBoardProfiling():

    def __init__(self):
        self.dataset = ChessBoardDataset(root_dir="../test/test_data",
                                         return_moves=True,
                                         return_outcome=True,
                                         transform=True,
                                         include_draws=False)
        self.n_test = 100

    @logger.catch
    def get_boards_indices(self):
        for i in range(self.n_test):
            indices = self.dataset.get_boards_indices(include_draws=True)

    @logger.catch
    def retrieve_board(self):
        for i in range(self.n_test):
            board, move_id, total_moves, result = self.dataset.retrieve_board(0)

    @logger.catch
    def getitems(self):
        for i in range(self.n_test):
            boards, moves, outcomes = self.dataset.__getitems__(list(range(64)))


if __name__ == "__main__":
    chess_profiling = ChessBoardProfiling()

    os.makedirs("./dataset", exist_ok=True)

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