import hashlib
from typing import Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from src.data.data_utils import dict_pieces, list_index_to_tensor
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
                              columns=self.columns)

        board_indexes = [data[piece] for key in dict_pieces for piece in dict_pieces[key]]

        t_board = torch.from_numpy(list_index_to_tensor(idxs=board_indexes))

        t_color = torch.tensor([data["active_color"][0]])
        t_castling = torch.tensor(data["castling"][0])

        outputs = {
            "board": t_board,
            "active_color": t_color,
            "castling": t_castling
        }

        if self.stockfish_eval:
            t_stockfish_eval = torch.tensor(data["stockfish_eval"][0])
            outputs["stockfish_eval"] = t_stockfish_eval

        if self.winner:
            t_winner = torch.tensor(data["winner"][0])
            outputs["winner"] = t_winner

        return outputs

    def __getitems__(self, indices: Union[Tensor, list[int]]) -> dict[str, Tensor]:
        """Returns the items at the given indices."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        indices = [self.indices[i] for i in indices]

        data = self.data.take(indices=indices,
                              columns=self.columns)

        board_indexes = [[data[piece][i] for key in dict_pieces for piece in dict_pieces[key]] for i in range(len(indices))]

        t_boards = torch.from_numpy(np.array([list_index_to_tensor(b) for b in board_indexes]))

        t_colors = torch.tensor([[c] for c in data["active_color"]])
        t_castlings = torch.tensor(data["castling"])

        outputs = {
            "board": t_boards,
            "active_color": t_colors,
            "castling": t_castlings
        }

        if self.stockfish_eval:
            t_stockfish_evals = torch.tensor(data["stockfish_eval"])
            outputs["stockfish_eval"] = t_stockfish_evals

        if self.winner:
            t_winners = torch.tensor(data["winner"])
            outputs["winner"] = t_winners

        return outputs

    def downsampling(self, seed: int = 42, ratio: float = 0.5) -> None:
        """Downsample the dataset.

        Args:
            seed (int): The seed for reproducibility.
            ratio (float): The ratio of downsampling.

        """
        np.random.seed(seed)

        new_size = int(len(self) * ratio)
        self.indices = np.random.choice(self.indices, size=new_size, replace=False)

        logger.info(f"Downsampled dataset to {new_size} samples.")
