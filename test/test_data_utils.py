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

        self.assertIsInstance(board, str)
        self.assertEqual(len(board), 64)

    def test_string_to_array(self):
        board = chess.Board()
        board = format_board(board)
        board = string_to_array(board)

        self.assertIsInstance(board, np.ndarray)
        self.assertEqual(board.shape, (6, 8, 8))

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
        assert np.array_equal(t_board, board_to_tensor(board)[0])

    def test_uci_to_coordinates(self):
        move = chess.Move.from_uci("e2e4")
        coordinates = uci_to_coordinates(move)

        self.assertIsInstance(coordinates, tuple)
        self.assertEqual(len(coordinates), 2)
        self.assertEqual(len(coordinates[0]), 2)
        self.assertEqual(len(coordinates[1]), 2)

        self.assertTrue(all(isinstance(x, int) for x in coordinates[0]))
        self.assertTrue(all(isinstance(x, int) for x in coordinates[1]))

        # from e2
        self.assertEqual(coordinates[0][0], 6)
        self.assertEqual(coordinates[0][1], 4)
        # to e4
        self.assertEqual(coordinates[1][0], 4)
        self.assertEqual(coordinates[1][1], 4)

    def test_moves_to_tensor(self):
        board = chess.Board()
        moves = list(board.legal_moves)
        tensor = moves_to_tensor(moves)

        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (64, 64))
        self.assertEqual(tensor.sum(), len(moves))
        self.assertEqual(tensor.dtype, np.int8)

    def test_board_to_tensor(self):
        board = chess.Board()
        indexes, active_color, castling = board_to_tensor(board)

        self.assertIsInstance(indexes, np.ndarray)
        self.assertEqual(indexes.shape, (12, 8, 8))
        self.assertEqual(indexes.sum(), 32)

        self.assertIsInstance(active_color, np.ndarray)
        self.assertEqual(active_color.shape, (1,))

        self.assertIsInstance(castling, np.ndarray)
        self.assertEqual(castling.shape, (4,))


    def test_result_to_tensor(self):
        result = "1-0"
        tensor = result_to_tensor(result)

        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1,))
        self.assertEqual(tensor.dtype, np.int8)

    def test_read_boards_from_pgn(self):
        boards = read_boards_from_pgn(pgn_file=self.pgn_path)

        self.assertIsInstance(boards, list)
        self.assertTrue(all(isinstance(board, chess.Board) for board in boards))
        self.assertEqual(len(boards), sum(self.len_moves_najdorf))
        self.assertTrue(all(board.is_valid() for board in boards))

        boards = read_boards_from_pgn(pgn_file=self.pgn_path,
                                      start_move=10,
                                      end_move=15)

        self.assertEqual(len(boards), sum([x - 25 for x in self.len_moves_najdorf]))

        boards = read_boards_from_pgn(pgn_file=self.pgn_path,
                                      start_move=35,
                                      end_move=35)
        self.assertEqual(len(boards), 20)
