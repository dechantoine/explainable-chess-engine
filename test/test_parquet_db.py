import os
import shutil
import unittest

import chess.pgn
import pyarrow.compute as pc

from src.data.parquet_db import ParquetChessDB, base_columns

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"


def lambda_func_board(boards: list[chess.Board]) -> list[float]:
    return [i for i in range(len(boards))]


class ParquetChessDBTestCase(unittest.TestCase):

    def setUp(self):
        self.list_index_read = [[53, 56],
                                None,
                                [36, 44],
                                [41],
                                [62],
                                [18, 42, 43, 45, 54, 55],
                                [6, 14],
                                [17],
                                [10],
                                [21],
                                [2],
                                [9, 23, 28],
                                0,
                                [0, 0, 0, 0],
                                -1,
                                2,
                                25]

        self.db = ParquetChessDB(test_db_dir)

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

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

    def test_len(self):
        self.db.add_directory(directory=test_data_dir)

        self.assertEqual(len(self.db), 1620)

    def test_get_columns(self):
        self.db.add_directory(directory=test_data_dir)
        columns = self.db.get_columns()

        self.assertCountEqual(columns, base_columns + ["winner", "game_id", "file_id"])


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
                                   full_move_number=25,
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
        assert all([len(i) == 17 for i in indexes])

    def test_take(self):
        self.db.add_directory(directory=test_data_dir)
        indexes = self.db.take(indices=[0, 345, 695, 1156, 1619])

        self.assertEqual(len(indexes), 5)
        assert all([len(i) == 20 for i in indexes])

        indexes = self.db.take(indices=[0, 345, 695, 1156, 1619],
                               columns=["winner"])
        self.assertEqual(len(indexes), 5)
        assert all([len(i) == 1 for i in indexes])
