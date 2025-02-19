import os
import shutil
import unittest

import torch

from src.data.parquet_dataset import ParquetChessDataset
from src.data.parquet_db import ParquetChessDB
from src.data.pgn_dataset import PGNDataset

test_data_dir = "test/test_data"
test_db_dir = "test/test_db"

class DatasetsTestCase(unittest.TestCase):
    def setUp(self):
        self.parquet_db = ParquetChessDB(test_db_dir)
        self.parquet_db.add_directory(directory=test_data_dir)

        self.parquet_dataset = ParquetChessDataset(path=test_db_dir,
                                                   stockfish_eval=False,
                                                   winner=False,
                                                   move_count=False)
        self.pgn_dataset = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=False,
            winner=False,
            move_count=False,
        )

    def tearDown(self):
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)

    def test_len(self):
        self.assertEqual(len(self.parquet_dataset), len(self.pgn_dataset))

    def test_getitems(self):
        boards_parquet = self.parquet_dataset.__getitems__(list(range(len(self.parquet_dataset))))
        boards_pgn = self.pgn_dataset.__getitems__(list(range(len(self.pgn_dataset))))

        torch.testing.assert_close(boards_parquet.board, boards_pgn.board)
