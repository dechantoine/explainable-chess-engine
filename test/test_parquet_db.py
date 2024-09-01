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
        self.dict_index_read = {"R": [53, 56],
                                "N": None,
                                "B": [36, 44],
                                "Q": [41],
                                "K": [62],
                                "P": [18, 42, 43, 45, 54, 55],
                                "r": [6, 14],
                                "n": [17],
                                "b": [10],
                                "q": [21],
                                "k": [2],
                                "p": [9, 23, 28],
                                "active_color": 0,
                                "castling": [0, 0, 0, 0],
                                "en_passant": -1,
                                "half_moves": 2,
                                "total_moves": 25}

        self.db = ParquetChessDB(test_db_dir)

        self.len_moves_najdorf = 266
        self.len_moves_tal = 507
        self.len_moves_morphy = 847

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

    def test_add_pgn(self):
        self.db.add_pgn(filepath=f"{test_data_dir}/Najdorf.pgn")

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet"))

        db = ParquetChessDB(test_db_dir)
        db.add_pgn(filepath=f"{test_data_dir}/Morphy.pgn")

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0_0.parquet"))

    def test_add_pgn_with_funcs(self):
        self.db.add_pgn(filepath=f"{test_data_dir}/Najdorf.pgn", funcs={"index": lambda_func_board})

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet"))

        values = self.db.read_boards(filters=[pc.field("file_id") == "Najdorf.pgn"],
                                     columns=["index"])

        self.assertEqual(values["index"], list(range(266)))

    def test_add_directory(self):
        self.db.add_directory(directory=test_data_dir)

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Tal.pgn/part_0_0.parquet"))

    def test_add_directory_with_funcs(self):
        self.db.add_directory(directory=test_data_dir, funcs={"index": lambda_func_board})

        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Morphy.pgn/part_0_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet"))
        self.assertTrue(os.path.exists(f"{test_db_dir}/file_id=Tal.pgn/part_0_0.parquet"))

        values = self.db.read_boards(filters=None,
                                     columns=["index"])

        self.assertEqual(values["index"],
                         list(range(self.len_moves_morphy)) +
                         list(range(self.len_moves_najdorf)) +
                         list(range(self.len_moves_tal)))

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

        self.assertCountEqual(files, [f"{test_db_dir}/file_id=Morphy.pgn/part_0_0.parquet",
                                      f"{test_db_dir}/file_id=Najdorf.pgn/part_0_0.parquet",
                                      f"{test_db_dir}/file_id=Tal.pgn/part_0_0.parquet"])

    def test_read_board(self):
        self.db.add_directory(directory=test_data_dir)
        board = self.db.read_board(file_id="Morphy.pgn",
                                   game_number=0,
                                   full_move_number=25,
                                   active_color=0)

        self.assertEqual(board, self.dict_index_read)

        board = self.db.read_board(file_id="Morphy.pgn",
                                   game_number=0,
                                   full_move_number=8,
                                   active_color=0,
                                   columns=["winner"])
        self.assertEqual(board["winner"], 1)

    def test_read_boards(self):
        self.db.add_directory(directory=test_data_dir)
        boards = self.db.read_boards(filters=[pc.field("active_color") == 0],
                                     columns=["winner"])

        self.assertEqual(len(boards["winner"]), 806)
        assert all([w in [-1, 0, 1] for w in boards["winner"]])

        boards = self.db.read_boards(filters=[pc.field("active_color") == 0,
                                              pc.field("winner") == 1],
                                     columns=None)

        self.assertEqual(len(boards), 17)
        assert all([len(i) == 547 for i in boards.values()])

    def test_take(self):
        self.db.add_directory(directory=test_data_dir)
        indexes = self.db.take(indices=[0, 345, 695, 1156, 1619])

        self.assertEqual(len(indexes), 20)
        assert all([len(i) == 5 for i in indexes.values()])

        indexes = self.db.take(indices=[0, 345, 695, 1156, 1619],
                               columns=["winner"])

        self.assertEqual(len(indexes), 1)
        assert all([len(i) == 5 for i in indexes.values()])

        indexes = self.db.take(indices=None,
                               columns=["winner"])

        self.assertEqual(len(indexes), 1)
        assert all([len(i) == 1620 for i in indexes.values()])
