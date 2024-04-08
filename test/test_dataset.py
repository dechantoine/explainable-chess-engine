from src.data.dataset import ChessBoardDataset

from chess import Board
from torch import Tensor
from loguru import logger

import unittest


class ChessBoardTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = ChessBoardDataset(root_dir="../test/test_data",
                                         return_moves=False,
                                         return_outcome=False,
                                         transform=False,
                                         include_draws=True)

    @logger.catch
    def test_get_boards_indices(self):
        indices = self.dataset.get_boards_indices(include_draws=True)
        assert isinstance(indices, list)
        assert all(isinstance(x, tuple) for x in indices)
        assert all(len(x) == 3 for x in indices)
        assert all(isinstance(x[i], int) for x in indices for i in range(3))

        self.dataset.return_outcome = True
        indices_no_draws = self.dataset.get_boards_indices(include_draws=False)

        _, results = self.dataset.__getitems__([indices.index(x) for x in indices_no_draws])
        assert len(indices) > len(indices_no_draws)
        assert all(result != "1/2-1/2" for result in results)

    @logger.catch
    def test_retrieve_board(self):
        board, move_id, total_moves, result = self.dataset.retrieve_board(0)
        assert isinstance(board, Board)
        assert isinstance(move_id, int)
        assert isinstance(total_moves, int)
        assert isinstance(result, str)

    @logger.catch
    def test_getitem(self):
        board = self.dataset[0]
        assert isinstance(board, Board)

        board = self.dataset[Tensor([0])]
        assert isinstance(board, Board)

        self.dataset.return_moves = True
        board, moves = self.dataset[0]
        assert isinstance(board, Board)
        assert isinstance(moves, list)

        self.dataset.return_outcome = True
        board, moves, outcome = self.dataset[0]
        assert isinstance(board, Board)
        assert isinstance(moves, list)
        assert isinstance(outcome, dict)
        assert(len(outcome) == 3)

        self.dataset.transform = True
        board, moves, outcome = self.dataset[0]
        assert isinstance(board, Tensor)
        assert isinstance(moves, Tensor)
        assert isinstance(outcome, Tensor)

        assert board.shape == (12, 8, 8)
        assert moves.shape == (64, 64)
        assert outcome.shape == (3,)
