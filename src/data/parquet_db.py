import os
from typing import Union

import chess.pgn
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from src.data.data_utils import arrays_to_lists, board_to_list_index, dict_pieces

base_columns = (list(dict_pieces["white"]) +
                list(dict_pieces["black"]) +
                ["active_color",
                 "castling",
                 "en_passant",
                 "half_moves",
                 "total_moves"])

base_fields = ([pa.field(name=piece, type=pa.list_(pa.int64())) for piece in list(dict_pieces["white"])] +
               [pa.field(name=piece, type=pa.list_(pa.int64())) for piece in list(dict_pieces["black"])]
               + [pa.field(name="active_color", type=pa.int64()),
                  pa.field(name="castling", type=pa.list_(pa.int64())),
                  pa.field(name="en_passant", type=pa.int64()),
                  pa.field(name="half_moves", type=pa.int64()),
                  pa.field(name="total_moves", type=pa.int64())])

PROCESSING_BATCH_SIZE = 1000


def and_filters(filters: list) -> pc.Expression:
    """Convert a list of filters to a pyarrow filter with an AND operation.

    Args:
        filters (list): list of filters.

    Returns:
        pc.Expression: pyarrow filter.

    """
    if not filters:
        return pc.scalar(True)

    elif len(filters) > 1:
        filter = filters[0]
        for f in filters[1:]:
            filter = filter & f
    else:
        filter = filters[0]

    return filter


def process_games_for_parquet(game: chess.pgn) -> tuple[pd.DataFrame, list[chess.Board]]:
    """Process a game for the parquet database.

    Args:
        game (chess.pgn): game to process.

    Returns:
        tuple: a tuple containing a dataframe and a list of boards.

    """
    boards = []

    board = game.board()
    boards.append(board.copy())
    idx_boards = [board_to_list_index(board)]

    for m in list(game.mainline_moves()):
        board.push(m)
        boards.append(board.copy())
        idxs = board_to_list_index(board)
        idx_boards.append(idxs)

    df_game = pd.DataFrame(idx_boards,
                           columns=base_columns)

    return df_game, boards


class ParquetChessDB:
    def __init__(self, path: str) -> None:
        """Initialize the ParquetChessDB object.

        Args:
            path (str): path to the directory containing the parquet files.

        """
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self._load()

    def __len__(self) -> int:
        return self.dataset.count_rows()

    def get_columns(self) -> list[str]:
        """Get the columns of the parquet database.

        Returns:
            list[str]: list of columns.

        """
        return self.dataset.schema.names

    def _load(self) -> None:
        """Load the parquet database."""
        self.dataset = ds.dataset(source=self.path,
                                  format="parquet",
                                  partitioning="hive")

    def add_pgn(self, filepath: str, funcs: dict = None) -> None:
        """Add a PGN file to the parquet database.

        Args:
            filepath (str): path to the PGN file.
            funcs (dict): dictionary of functions to apply to each board with the key being the column name in the parquet
             file.

        """
        if not funcs:
            funcs = {}

        pgn = open(filepath)
        file_id = os.path.basename(filepath)
        k = 0
        game_id = 0

        game = True

        boards = []
        df = pd.DataFrame(columns=base_columns
                                  + ["winner", "game_id", "file_id"])

        while game:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception as e:
                logger.error(f"Error reading game: {e}")
                continue

            if not game:
                for col, func in funcs.items():
                    df[col] = func(boards)

                pq.write_to_dataset(table=pa.Table.from_pandas(df=df,
                                                               preserve_index=False),
                                    schema=pa.schema(fields=base_fields +
                                                            [pa.field(name="winner", type=pa.int64()),
                                                             pa.field(name="game_id", type=pa.int64()),
                                                             pa.field(name="file_id", type=pa.string())] +
                                                            [pa.field(name=col, type=pa.infer_type(values=df[col]))
                                                             for col in funcs.keys()]),
                                    root_path=self.path,
                                    partition_cols=["file_id"],
                                    basename_template="part_{{i}}_{batch_id}.parquet".format(batch_id=k))

                break

            winner = 1 if game.headers["Result"] == "1-0" else 0 if game.headers["Result"] == "0-1" else -1

            df_game, game_boards = process_games_for_parquet(game)

            df_game["winner"] = winner
            df_game["game_id"] = game_id
            df_game["file_id"] = file_id

            df = pd.concat(objs=[df, df_game], ignore_index=True)
            boards += game_boards

            if (game_id + 1) % PROCESSING_BATCH_SIZE == 0:

                for col, func in funcs.items():
                    df[col] = func(boards)

                pq.write_to_dataset(table=pa.Table.from_pandas(df=df,
                                                               preserve_index=False),
                                    schema=pa.schema(fields=base_fields +
                                                            [pa.field(name="winner", type=pa.int64()),
                                                             pa.field(name="game_id", type=pa.int64()),
                                                             pa.field(name="file_id", type=pa.string())] +
                                                            [pa.field(name=col, type=pa.infer_type(values=df[col]))
                                                             for col in funcs.keys()]),
                                    root_path=self.path,
                                    partition_cols=["file_id"],
                                    basename_template="part_{{i}}_{batch_id}.parquet".format(batch_id=k))

                k += 1
                boards = []
                df = pd.DataFrame(columns=base_columns
                                          + ["winner", "game_id", "file_id"])

                self._load()
                return

            game_id += 1

        self._load()

    def add_directory(self, directory: str, funcs: dict = None) -> None:
        """Add a directory of PGN files to the parquet database.

        Args:
            directory (str): path to the directory containing the PGN files.
            funcs (dict): dictionary of functions to apply to each board with the key being the column name in the parquet
        file.

        """
        logger.info("Adding directory to ParquetChessDB...")

        if not funcs:
            funcs = {}

        dir = os.listdir(directory)
        dir.sort()

        logger.info(f"Found {len(dir)} files in the directory.")

        existing_files = self.list_files()
        existing_files = [file.split("file_id=")[1].split("/")[0] for file in existing_files]

        logger.info(f"{len(existing_files)} files already in the database.")

        dir = [file for file in dir if file.endswith(".pgn") and file not in existing_files]

        for file in tqdm(iterable=dir,
                         desc="Processing PGN files..."):
            file_path = os.path.join(directory, file)
            self.add_pgn(filepath=file_path,
                         funcs=funcs)

        if len(existing_files) > 0:
            logger.info(f"Added {len(dir)} files to the database.")
        else:
            logger.info("Directory added to the new database.")

    def list_files(self) -> list[str]:
        """List the files in the parquet database.

        Returns:
            list[str]: list of files.

        """
        files = self.dataset.files.copy()
        files.sort()
        return files

    def schema(self) -> pa.Schema:
        """Get the schema of the parquet database.

        Returns:
            pa.Schema: schema of the parquet database.

        """
        return self.dataset.schema

    def read_board(self, file_id: str, game_number: int = 0, full_move_number: int = 0, active_color: int = 0,
                   columns: list = None) -> dict:
        """Read a unique board from the parquet database by file id, game number, full move number, and active color.

        Args:
            file_id (str): file id.
            game_number (int): game number.
            full_move_number (int): full move number.
            active_color (int): active color.
            columns (list): columns to read. Default to all columns.

        Returns:
            dict: dictionary of desired columns.

        """
        if not columns:
            columns = base_columns
        table = self.dataset.to_table(columns=columns,
                                      filter=and_filters([pc.field("file_id") == file_id,
                                                          pc.field("game_id") == game_number,
                                                          pc.field("active_color") == active_color,
                                                          pc.field("total_moves") == full_move_number])
                                      )
        indexes = arrays_to_lists(data=table.to_pandas().values[0])

        outputs = {col: indexes[i] for i, col in enumerate(columns)}

        return outputs

    def read_boards(self, filters: list = None, columns: list = None) -> dict:
        """Read boards from the parquet database with filters.

        Args:
            filters (list): filters to apply.
            columns (list): columns to read. Default to all columns.

        Returns:
            dict: dictionary of desired columns.

        """
        if not columns:
            columns = base_columns

        table = self.dataset.to_table(columns=columns,
                                      filter=and_filters(filters)
                                      )
        indexes = arrays_to_lists(data=table.to_pandas().values)

        outputs = {col: [indexes[i][j] for i in range(len(indexes))] for j, col in enumerate(columns)}

        return outputs

    def take(self, indices: Union[int, list[int]] = None, columns: list[str] = None) -> dict:
        """Read boards from the parquet database by indices.

        Args:
            indices (Union[int, list[int]]): indices to read. Default to all indices.
            columns (list[str]): columns to read.

        Returns:
            dict: dictionary of desired columns.

        """
        if not indices:
            return self.read_boards(columns=columns)

        if isinstance(indices, int):
            indices = [indices]

        if not columns:
            columns = self.get_columns()

        indexes = arrays_to_lists(
            data=self.dataset.take(
                indices=indices,
                columns=columns
            ).to_pandas().values)

        outputs = {col: [indexes[i][j] for i in range(len(indexes))] for j, col in enumerate(columns)}

        return outputs
