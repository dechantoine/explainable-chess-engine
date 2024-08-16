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


def process_pgn_for_parquet(filepath: str) -> tuple[pd.DataFrame, list[chess.Board]]:
    """Process a PGN file for the parquet database.

    Args:
        filepath (str): path to the PGN file.

    Returns:
        tuple: a tuple containing a dataframe and a list of boards.

    """
    pgn = open(filepath)
    file_id = os.path.basename(filepath)

    game_id = 0
    boards = []
    df = pd.DataFrame(columns=base_columns
                              + ["winner", "game_id", "file_id"])

    while game := chess.pgn.read_game(pgn):
        winner = 1 if game.headers["Result"] == "1-0" else 0 if game.headers["Result"] == "0-1" else -1

        df_game, game_boards = process_games_for_parquet(game)

        df_game["winner"] = winner
        df_game["game_id"] = game_id
        df_game["file_id"] = file_id

        df = pd.concat(objs=[df, df_game], ignore_index=True)
        boards += game_boards

        game_id += 1

    return df, boards


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
        logger.info("Adding PGN file to ParquetChessDB...")

        if not funcs:
            funcs = {}

        df, boards = process_pgn_for_parquet(filepath)

        for col, func in funcs.items():
            logger.info(f"Applying function {col} to boards.")
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
                            basename_template="part_{i}.parquet")

        self._load()

        logger.info("PGN file added.")

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

        df = pd.DataFrame(columns=base_columns
                                  + ["winner", "game_id", "file_id"])
        boards = []

        dir = os.listdir(directory)
        dir.sort()
        for file in tqdm(iterable=dir,
                         desc="Processing PGN files..."):
            if file.endswith(".pgn"):
                file_path = os.path.join(directory, file)
                df_file, boards_file = process_pgn_for_parquet(file_path)
                df = pd.concat(objs=[df, df_file], ignore_index=True)
                boards += boards_file

        for col, func in funcs.items():
            logger.info(f"Applying function {col} to boards.")
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
                            basename_template="part_{i}.parquet")

        self._load()

        logger.info("Directory added.")

    def list_files(self) -> list[str]:
        """List the files in the parquet database.

        Returns:
            list[str]: list of files.

        """
        return self.dataset.files

    def schema(self) -> pa.Schema:
        """Get the schema of the parquet database.

        Returns:
            pa.Schema: schema of the parquet database.

        """
        return self.dataset.schema

    def read_board(self, file_id: str, game_number: int = 0, full_move_number: int = 0, active_color: int = 0,
                   columns: list = None) -> list:
        """Read a unique board from the parquet database by file id, game number, full move number, and active color.

        Args:
            file_id (str): file id.
            game_number (int): game number.
            full_move_number (int): full move number.
            active_color (int): active color.
            columns (list): columns to read. Default to all columns.

        Returns:
            list: list of indexes.

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

        return indexes

    def read_boards(self, filters: list = None, columns: list = None) -> list:
        """Read boards from the parquet database with filters.

        Args:
            filters (list): filters to apply.
            columns (list): columns to read. Default to all columns.

        Returns:
            list: list of lists of indexes.

        """
        if not columns:
            columns = base_columns

        table = self.dataset.to_table(columns=columns,
                                      filter=and_filters(filters)
                                      )
        indexes = arrays_to_lists(data=table.to_pandas().values)

        return indexes

    def take(self, indices: Union[int, list[int]], columns: list[str] = None) -> list:
        """Read boards from the parquet database by indices.

        Args:
            indices (Union[int, list[int]]): indices to read.
            columns (list[str]): columns to read.

        Returns:
            list: list of lists of indexes.

        """
        if isinstance(indices, int):
            indices = [indices]

        indexes = arrays_to_lists(
            data=self.dataset.take(
                indices=indices,
                columns=columns
            ).to_pandas().values)

        return indexes
