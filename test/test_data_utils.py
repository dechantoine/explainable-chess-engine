import unittest

import chess
import numpy as np

from src.data.data_utils import (
    board_to_list_index,
    board_to_tensor,
    format_board,
    list_index_to_fen,
    list_index_to_tensor,
    moves_to_tensor,
    read_boards_from_pgn,
    result_to_tensor,
    string_to_array,
    uci_to_coordinates,
)


class DataUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.pgn_path = "test/test_data/Najdorf.pgn"
        self.len_moves_najdorf = [73, 58, 44, 87]
        self.fen = "6k1/5ppp/3qp3/p2n4/P1pP4/1rP3P1/5P1P/RNQ3K1 b - - 4 28"
        self.list_index = [[56],
                           [57],
                           None,
                           [58],
                           [62],
                           [32, 35, 42, 46, 53, 55],
                           [41],
                           [27],
                           None,
                           [19],
                           [6],
                           [13, 14, 15, 20, 24, 34],
                           0,
                           [0, 0, 0, 0],
                           -1,
                           4,
                           28]

    def test_format_board(self):
        board = chess.Board()
        board = format_board(board)

        assert isinstance(board, str)
        assert len(board) == 64

    def test_string_to_array(self):
        board = chess.Board()
        board = format_board(board)
        board = string_to_array(board)

        assert isinstance(board, np.ndarray)
        assert board.shape == (6, 8, 8)

    def test_board_to_list_index(self):
        board = chess.Board(fen=self.fen)
        list_index = board_to_list_index(board)

        self.assertEqual(list_index, self.list_index)

    def test_list_index_to_fen(self):
        fen = list_index_to_fen(self.list_index)

        self.assertEqual(fen, self.fen)

    def test_list_index_to_tensor(self):
        t_board = list_index_to_tensor(self.list_index)

        self.assertIsInstance(t_board, np.ndarray)

        self.assertEqual(t_board.shape, (12, 8, 8))

        board = chess.Board(fen=self.fen)
        assert np.array_equal(t_board, board_to_tensor(board))

    def test_uci_to_coordinates(self):
        move = chess.Move.from_uci("e2e4")
        coordinates = uci_to_coordinates(move)

        assert isinstance(coordinates, tuple)
        assert len(coordinates) == 2
        assert len(coordinates[0]) == 2
        assert len(coordinates[1]) == 2

        assert all(isinstance(x, int) for x in coordinates[0])
        assert all(isinstance(x, int) for x in coordinates[1])

        # from e2
        assert coordinates[0][0] == 6
        assert coordinates[0][1] == 4
        # to e4
        assert coordinates[1][0] == 4
        assert coordinates[1][1] == 4

    def test_moves_to_tensor(self):
        board = chess.Board()
        moves = list(board.legal_moves)
        tensor = moves_to_tensor(moves)

        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (64, 64)
        assert tensor.sum() == len(moves)
        assert tensor.dtype == np.int8

    def test_board_to_tensor(self):
        board = chess.Board()
        tensor = board_to_tensor(board)

        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (12, 8, 8)
        assert tensor.sum() == 32
        assert tensor.dtype == np.int8

    def test_result_to_tensor(self):
        result = "1-0"
        tensor = result_to_tensor(result)

        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1,)
        assert tensor.dtype == np.int8

    def test_read_boards_from_pgn(self):
        boards = read_boards_from_pgn(pgn_file=self.pgn_path)

        assert isinstance(boards, list)
        assert all(isinstance(board, chess.Board) for board in boards)
        assert len(boards) == sum(self.len_moves_najdorf)
        assert all(board.is_valid() for board in boards)

        boards = read_boards_from_pgn(pgn_file=self.pgn_path,
                                      start_move=10,
                                      end_move=15)

        assert len(boards) == sum([x - 25 for x in self.len_moves_najdorf])

        boards = read_boards_from_pgn(pgn_file=self.pgn_path,
                                      start_move=35,
                                      end_move=35)
        assert len(boards) == 20
