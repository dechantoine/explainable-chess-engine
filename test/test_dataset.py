import unittest

import numpy as np
from chess import Board
from torch import Tensor

from src.data.dataset import ChessBoardDataset


class ChessBoardTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=False,
            return_outcome=False,
            transform=False,
            include_draws=True,
            in_memory=False,
        )

        self.dataset_in_memory = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=False,
            return_outcome=False,
            transform=False,
            include_draws=True,
            in_memory=True,
            num_workers=8,
        )

        self.dataset_in_memory_return_moves = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=True,
            return_outcome=False,
            transform=False,
            include_draws=True,
            in_memory=True,
            num_workers=8,
        )

        self.dataset_in_memory_return_outcome = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=False,
            return_outcome=True,
            transform=False,
            include_draws=True,
            in_memory=True,
            num_workers=8,
        )

        self.dataset_in_memory_return_both = ChessBoardDataset(
            root_dir="../test/test_data",
            return_moves=True,
            return_outcome=True,
            transform=False,
            include_draws=True,
            in_memory=True,
            num_workers=8,
        )

        self.len_moves_najdorf = [73, 58, 44, 87]
        self.len_moves_tal = [60, 127, 82, 59, 45, 128]
        self.len_moves_morphy = [
            61,
            35,
            0,
            33,
            46,
            45,
            21,
            45,
            29,
            41,
            91,
            39,
            56,
            39,
            109,
            27,
            47,
            28,
        ]

        self.results_najdorf = [0, 0, -1, 0, 0]
        self.results_tal = [-1, 1, 0, 1, 1, 1]
        self.results_morphy = [1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]

        self.len_moves_najdorf_no_draws = [44]
        self.len_moves_tal_no_draws = [60, 127, 59, 45, 128]
        self.len_moves_morphy_no_draws = [
            61,
            35,
            0,
            33,
            46,
            45,
            21,
            45,
            29,
            41,
            91,
            39,
            56,
            39,
            109,
            27,
            47,
            28,
        ]

    def test_init(self):
        self.assertRaises(AttributeError, lambda: self.dataset.board_samples)
        self.assertRaises(AttributeError, lambda: self.dataset.legal_moves_samples)
        self.assertRaises(AttributeError, lambda: self.dataset.outcomes)

        assert self.dataset_in_memory.board_samples is not None
        self.assertRaises(
            AttributeError, lambda: self.dataset_in_memory.legal_moves_samples
        )
        self.assertRaises(AttributeError, lambda: self.dataset_in_memory.outcomes)

        assert self.dataset_in_memory_return_moves.board_samples is not None
        assert self.dataset_in_memory_return_moves.legal_moves_samples is not None
        self.assertRaises(
            AttributeError, lambda: self.dataset_in_memory_return_moves.outcomes
        )

        assert self.dataset_in_memory_return_outcome.board_samples is not None
        self.assertRaises(
            AttributeError,
            lambda: self.dataset_in_memory_return_outcome.legal_moves_samples,
        )
        assert self.dataset_in_memory_return_outcome.outcomes is not None

        assert self.dataset_in_memory_return_both.board_samples is not None
        assert self.dataset_in_memory_return_both.legal_moves_samples is not None
        assert self.dataset_in_memory_return_both.outcomes is not None

    def test_get_boards_indices(self):
        for d in [
            self.dataset,
            self.dataset_in_memory,
            self.dataset_in_memory_return_moves,
            self.dataset_in_memory_return_outcome,
            self.dataset_in_memory_return_both,
        ]:
            indices, results = d.get_boards_indices(include_draws=True)
            assert isinstance(indices, list)
            assert all(isinstance(x, tuple) for x in indices)
            assert all(len(x) == 3 for x in indices)
            assert all(isinstance(x[i], int) for x in indices for i in range(3))

            assert len([x for x in indices if x[0] == 0]) == sum(self.len_moves_najdorf)
            assert len([x for x in indices if x[0] == 1]) == sum(self.len_moves_tal)
            assert len([x for x in indices if x[0] == 2]) == sum(self.len_moves_morphy)

            assert all(
                len([x for x in indices if x[0] == 0 and x[1] == i])
                == self.len_moves_najdorf[i]
                for i in range(len(self.len_moves_najdorf))
            )
            assert all(
                len([x for x in indices if x[0] == 1 and x[1] == i])
                == self.len_moves_tal[i]
                for i in range(len(self.len_moves_tal))
            )
            assert all(
                len([x for x in indices if x[0] == 2 and x[1] == i])
                == self.len_moves_morphy[i]
                for i in range(len(self.len_moves_morphy))
            )

            assert isinstance(results, list)
            assert all(isinstance(x, np.int8) for x in results)
            assert len(results) == len(indices)
            assert all(
                results[sum(self.len_moves_najdorf[:i]) + k] == self.results_najdorf[i]
                for i in range(len(self.len_moves_najdorf))
                for k in range(self.len_moves_najdorf[i])
            )
            assert all(
                results[sum(self.len_moves_najdorf) + sum(self.len_moves_tal[:i]) + k]
                == self.results_tal[i]
                for i in range(len(self.len_moves_tal))
                for k in range(self.len_moves_tal[i])
            )
            assert all(
                results[
                    sum(self.len_moves_najdorf)
                    + sum(self.len_moves_tal)
                    + sum(self.len_moves_morphy[:i])
                    + k
                ]
                == self.results_morphy[i]
                for i in range(len(self.len_moves_morphy))
                for k in range(self.len_moves_morphy[i])
            )

            indices_no_draws, results_no_draws = d.get_boards_indices(
                include_draws=False
            )
            assert len([x for x in indices_no_draws if x[0] == 0]) == sum(
                self.len_moves_najdorf_no_draws
            )
            assert len([x for x in indices_no_draws if x[0] == 1]) == sum(
                self.len_moves_tal_no_draws
            )
            assert len([x for x in indices_no_draws if x[0] == 2]) == sum(
                self.len_moves_morphy_no_draws
            )

    def test_retrieve_board(self):
        for d in [
            self.dataset,
            self.dataset_in_memory,
            self.dataset_in_memory_return_moves,
            self.dataset_in_memory_return_outcome,
            self.dataset_in_memory_return_both,
        ]:
            board, move_id, total_moves, result = d.retrieve_board(2)
            assert isinstance(move_id, int)
            assert isinstance(total_moves, int)
            assert isinstance(result, str)

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
        assert len(outcome) == 3

        self.dataset.transform = True
        for d in [self.dataset, self.dataset_in_memory_return_both]:
            d.transform = True
            board, moves, outcome = d[0]
            assert isinstance(board, Tensor)
            assert isinstance(moves, Tensor)
            assert isinstance(outcome, Tensor)

            assert board.shape == (12, 8, 8)
            assert moves.shape == (64, 64)
            assert outcome.shape == (3,)

        board = self.dataset_in_memory[Tensor([0])]
        assert isinstance(board, Tensor)
        assert board.shape == (12, 8, 8)

        board, moves = self.dataset_in_memory_return_moves[Tensor([0])]
        assert isinstance(board, Tensor)
        assert isinstance(moves, Tensor)
        assert board.shape == (12, 8, 8)
        assert moves.shape == (64, 64)

        board, outcome = self.dataset_in_memory_return_outcome[Tensor([0])]
        assert isinstance(board, Tensor)
        assert isinstance(outcome, Tensor)
        assert board.shape == (12, 8, 8)
        assert outcome.shape == (3,)

    def test_get_items(self):
        boards = self.dataset.__getitems__([0, 1, 2])
        assert isinstance(boards, list)
        assert all(isinstance(b, Board) for b in boards)

        boards = self.dataset.__getitems__(Tensor([0, 1, 2]))
        assert isinstance(boards, list)
        assert all(isinstance(b, Board) for b in boards)

        self.dataset.return_moves = True
        boards, moves = self.dataset.__getitems__([0, 1, 2])
        assert isinstance(boards, list)
        assert all(isinstance(b, Board) for b in boards)
        assert isinstance(moves, list)
        assert all(isinstance(m, list) for m in moves)

        self.dataset.return_outcome = True
        boards, moves, outcomes = self.dataset.__getitems__([0, 1, 2])
        assert isinstance(boards, list)
        assert all(isinstance(b, Board) for b in boards)
        assert isinstance(moves, list)
        assert all(isinstance(m, list) for m in moves)
        assert isinstance(outcomes, list)
        assert all(isinstance(o, dict) for o in outcomes)

        self.dataset.transform = True
        for d in [self.dataset, self.dataset_in_memory_return_both]:
            boards, moves, outcomes = d.__getitems__([0, 1, 2])
            assert isinstance(boards, Tensor)
            assert all(isinstance(b, Tensor) for b in boards)
            assert isinstance(moves, Tensor)
            assert all(isinstance(m, Tensor) for m in moves)
            assert isinstance(outcomes, Tensor)
            assert all(isinstance(o, Tensor) for o in outcomes)

            assert all(b.shape == (12, 8, 8) for b in boards)
            assert all(m.shape == (64, 64) for m in moves)
            assert all(o.shape == (3,) for o in outcomes)

        boards = self.dataset_in_memory.__getitems__([0, 1, 2])
        assert isinstance(boards, Tensor)
        assert all(isinstance(b, Tensor) for b in boards)
        assert all(b.shape == (12, 8, 8) for b in boards)

        boards, moves = self.dataset_in_memory_return_moves.__getitems__([0, 1, 2])
        assert isinstance(boards, Tensor)
        assert all(isinstance(b, Tensor) for b in boards)
        assert all(b.shape == (12, 8, 8) for b in boards)
        assert isinstance(moves, Tensor)
        assert all(isinstance(m, Tensor) for m in moves)
        assert all(m.shape == (64, 64) for m in moves)

        boards, outcomes = self.dataset_in_memory_return_outcome.__getitems__([0, 1, 2])
        assert isinstance(boards, Tensor)
        assert all(isinstance(b, Tensor) for b in boards)
        assert all(b.shape == (12, 8, 8) for b in boards)
        assert isinstance(outcomes, Tensor)
        assert all(isinstance(o, Tensor) for o in outcomes)
        assert all(o.shape == (3,) for o in outcomes)
