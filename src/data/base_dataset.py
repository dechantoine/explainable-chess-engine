from abc import ABC
from typing import Optional, Union

from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset


class BoardItem(BaseModel):
    board: Tensor
    active_color: Tensor
    castling: Tensor
    winner: Optional[Tensor] = None
    move_id: Optional[Tensor] = None
    total_moves: Optional[Tensor] = None
    stockfish_eval: Optional[Tensor] = None

    class Config:
        arbitrary_types_allowed = True


class BaseChessDataset(ABC, Dataset):
    def __init__(self,
                 stockfish_eval: bool = False,
                 winner: bool = False,
                 move_count: bool = False):
        """Initializes the BaseChessDataset class.

        Args:
            stockfish_eval (bool): Whether to include Stockfish evaluations in outputs.
            winner (bool): Whether to include winner in outputs.
            move_count (bool): Whether to include move count in outputs.

        """
        self.stockfish_eval = stockfish_eval
        self.winner = winner
        self.move_count = move_count

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        pass

    def __getitem__(self, idx: Union[Tensor, int]) -> BoardItem:
        """Returns the item at the given index."""
        pass

    def __getitems__(self, indices: Union[Tensor, list[int]]) -> BoardItem:
        """Returns the items at the given indices."""
        pass
