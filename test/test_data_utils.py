from src.data.data_utils import (format_board,
                                 string_to_array,
                                 uci_to_coordinates,
                                 moves_to_tensor,
                                 board_to_tensor,
                                 result_to_tensor)
import chess

import numpy as np
from loguru import logger

import unittest


class DataUtilsTestCase(unittest.TestCase):
    @logger.catch
    def test_format_board(self):
        board = chess.Board()
        board = format_board(board)
        assert isinstance(board, str)
        assert len(board) == 64


    @logger.catch
    def test_string_to_array(self):
        board = chess.Board()
        board = format_board(board)
        board = string_to_array(board)
        assert isinstance(board, np.ndarray)
        assert board.shape == (6, 8, 8)


    @logger.catch
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


    @logger.catch
    def test_moves_to_tensor(self):
        board = chess.Board()
        moves = list(board.legal_moves)
        tensor = moves_to_tensor(moves)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (64, 64)
        assert tensor.sum() == len(moves)
        assert tensor.dtype == np.int8


    @logger.catch
    def test_board_to_tensor(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (12, 8, 8)
        assert tensor.sum() == 32
        assert tensor.dtype == np.int8


    @logger.catch
    def test_result_to_tensor(self):
        result = "1-0"
        tensor = result_to_tensor(result)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (1,)
        assert tensor.dtype == np.int8