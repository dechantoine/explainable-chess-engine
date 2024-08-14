import os
import shutil
import unittest

import chess.pgn
import pyarrow.compute as pc

from src.data.parquet_db import ParquetChessDB, board_to_list_index, list_index_to_fen

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"


def lambda_func_board(boards: list[chess.Board]) -> list[float]:
    return [i for i in range(len(boards))]


class ParquetChessDBTestCase(unittest.TestCase):

    def setUp(self):
        self.fen = "rn1q1rk1/pb1pbppp/1p2pn2/2p5/3P4/2P2NP1/PP1NPPBP/R1BQR1K1 b - - 5 8"
        self.list_index = [[[56, 60], [45, 51], [54, 58], [59], [62], [35, 42, 46, 48, 49, 52, 53, 55]],
                           [[0, 5], [1, 21], [9, 12], [3], [6], [8, 11, 13, 14, 15, 17, 20, 26]],
                           0,
                           [0, 0, 0, 0],
                           -1,
                           5,
                           8]
        self.list_index_read = [[[56, 63], [45], [34, 58], [59], [60], [43, 48, 49, 50, 53, 54, 55]],
                                [[0, 7], [1], [26, 29], [21], [4], [8, 9, 10, 14, 15, 28]],
                                0,
                                [1, 1, 1, 1],
                                -1,
                                1,
                                8]
        self.board = chess.Board(fen=self.fen)

        self.db = ParquetChessDB(test_db_dir)

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

    def test_board_to_list_index(self):
        list_index = board_to_list_index(self.board)

        self.assertEqual(list_index, self.list_index)

    def test_list_index_to_fen(self):
        fen = list_index_to_fen(self.list_index)

        self.assertEqual(fen, self.fen)

    def test_add_pgn(self):
        self.db.add_pgn(filepath=f"{test_data_dir}/Najdorf.pgn")

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet"))

        db = ParquetChessDB(test_db_dir)
        db.add_pgn(filepath=f"{test_data_dir}/Morphy.pgn")

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0.parquet"))

    def test_add_pgn_with_funcs(self):
        self.db.add_pgn(filepath=f"{test_data_dir}/Najdorf.pgn", funcs={"index": lambda_func_board})

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet"))

        values = self.db.read_boards(filters=[pc.field("file_id") == "Najdorf.pgn"],
                                     columns=["index"])

        self.assertEqual(values, [[i] for i in range(266)])

    def test_add_directory(self):
        self.db.add_directory(directory=test_data_dir)

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Tal.pgn/part_0.parquet"))

    def test_add_directory_with_funcs(self):
        self.db.add_directory(directory=test_data_dir, funcs={"index": lambda_func_board})

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Tal.pgn/part_0.parquet"))

        values = self.db.read_boards(filters=None,
                                     columns=["index"])

        self.assertEqual(values, [[i] for i in range(1620)])
    def test_list_files(self):
        self.db.add_directory(directory=test_data_dir)
        files = self.db.list_files()

        self.assertCountEqual(files, [f"{test_db_dir}/file_id=Morphy.pgn/part_0.parquet",
                                      f"{test_db_dir}/file_id=Najdorf.pgn/part_0.parquet",
                                      f"{test_db_dir}/file_id=Tal.pgn/part_0.parquet"])

    def test_read_board(self):
        self.db.add_directory(directory=test_data_dir)
        board = self.db.read_board(file_id="Morphy.pgn",
                                   game_number=0,
                                   full_move_number=8,
                                   active_color=0)

        self.assertEqual(board, self.list_index_read)

        board = self.db.read_board(file_id="Morphy.pgn",
                                   game_number=0,
                                   full_move_number=8,
                                   active_color=0,
                                   columns=["winner"])
        self.assertEqual(board, [1])

    def test_read_boards(self):
        self.db.add_directory(directory=test_data_dir)
        winners = self.db.read_boards(filters=[pc.field("active_color") == 0],
                                      columns=["winner"])

        self.assertEqual(len(winners), 806)
        assert all([len(w) == 1 for w in winners])
        assert all([w[0] in [-1, 0, 1] for w in winners])

        indexes = self.db.read_boards(filters=[pc.field("active_color") == 0,
                                               pc.field("winner") == 1],
                                      columns=None)
        self.assertEqual(len(indexes), 547)
        assert all([len(i) == 7 for i in indexes])
        assert all([len(i[0]) == 6 for i in indexes])
        assert all([len(i[1]) == 6 for i in indexes])
        assert all([i[2] in [-1, 0, 1] for i in indexes])
        assert all([len(i[3]) == 4 for i in indexes])
