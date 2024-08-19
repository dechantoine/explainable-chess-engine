import hashlib
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.data_utils import list_index_to_tensor
from src.data.parquet_db import ParquetChessDB, base_columns


class ParquetChessDataset(Dataset):
    """Dataset for the Parquet Chess DB."""

    def __init__(self,
                 path: str,
                 stockfish_eval: bool = True,
                 winner: bool = False
                 ) -> None:
        """Initializes the ParquetChessDataset class.

        Args:
            path (str): The path to the Parquet file.
            stockfish_eval (bool): Whether to include Stockfish evaluations in outputs.
            winner (bool): Whether to include winner in outputs.

        """
        self.data = ParquetChessDB(path)
        self.indices = np.arange(len(self.data))
        self.set_columns(stockfish_eval=stockfish_eval, winner=winner)

    def set_columns(self, stockfish_eval: bool = True, winner: bool = False) -> None:
        """Sets the columns to include in the dataset outputs.

        Args:
            stockfish_eval (bool): Whether to include Stockfish evaluations in outputs.
            winner (bool): Whether to include winner in outputs.

        """
        self.stockfish_eval = stockfish_eval
        self.winner = winner

        self.columns = base_columns.copy()
        self.columns.remove("en_passant")
        self.columns.remove("half_moves")
        self.columns.remove("total_moves")

        if stockfish_eval:
            self.columns.append("stockfish_eval")

        if winner:
            self.columns.append("winner")

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.indices)

    def get_hash(self) -> str:
        """Get the hash of the dataset."""
        return hashlib.md5(
            (str(self.data.list_files()) + str(self.data.schema) + str(self.indices)).encode()
        ).hexdigest()

    def __getitem__(self, idx: Union[Tensor, int]) -> dict[str, Tensor]:
        """Returns the item at the given index."""
        if isinstance(idx, Tensor):
            idx = idx.item()

        idx = self.indices[idx]

        data = self.data.take(indices=[idx],
                              columns=self.columns)[0]

        board_indexes = data[:12]

        t_indexes = torch.from_numpy(list_index_to_tensor(idxs=board_indexes))
        t_color = torch.tensor(data[12])
        t_castling = torch.tensor(data[13])

        if self.stockfish_eval:
            t_stockfish_eval = torch.tensor(data[14])
            return t_indexes, t_color, t_castling, t_stockfish_eval

        return t_indexes, t_color, t_castling

    def __getitems__(self, indices: Union[Tensor, list[int]]) -> Union[
        tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Returns the items at the given indices."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        data = self.data.take(indices=indices,
                              columns=self.columns)

        t_indexes = torch.from_numpy(np.array([list_index_to_tensor(idxs=d[:12]) for d in data]))
        t_color = torch.tensor([d[12] for d in data])
        t_castling = torch.tensor([d[13] for d in data])

        if self.stockfish_eval:
            t_stockfish_eval = torch.tensor([d[14] for d in data])
            return t_indexes, t_color, t_castling, t_stockfish_eval

        return t_indexes, t_color, t_castling
