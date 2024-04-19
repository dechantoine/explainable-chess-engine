import os
from typing import Union

import chess.pgn
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from src.data.data_utils import (board_to_tensor, moves_to_tensor, result_to_tensor,
                                 batch_boards_to_tensor, batch_moves_to_tensor, batch_results_to_tensor)
from loguru import logger


class ChessBoardDataset(Dataset):
    """Chess boards with legal moves dataset."""

    @logger.catch
    def __init__(self,
                 root_dir: str,
                 return_moves: bool = False,
                 return_outcome: bool = False,
                 transform: bool = False,
                 include_draws: bool = False
                 ):
        """
        Arguments:
            root_dir (string): Directory with all the PGNs.
            return_moves (bool): Return the legal moves for each board.
            return_outcome (bool): Return the outcome of the game for each board.
            transform (bool): Apply the transform to the boards and legal moves.
            include_draws (bool): Include draws in the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_moves = return_moves
        self.return_outcome = return_outcome
        self.list_pgn_files = [f for f in os.listdir(self.root_dir) if f.endswith(".pgn")]
        self.board_indices = self.get_boards_indices(include_draws=include_draws)

    @logger.catch
    def get_boards_indices(self,
                           include_draws: bool = False,
                           ) -> list[tuple[int, int, int]]:
        """Get the indices of all the boards in the dataset.

        Args:
            include_draws (bool): Include draws in the dataset.

        Returns:
            list[tuple[int, int, int]]: List of tuples containing the file index, game index, and move index.
        """
        list_board_indices = []
        for i, file in enumerate(self.list_pgn_files):
            pgn = open(self.root_dir + "/" + file)
            j = -1

            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                j += 1

                try:
                    result = result_to_tensor(game.headers["Result"])
                except ValueError:
                    continue
                else:
                    if not include_draws and result == np.array([0], dtype=np.int8):
                        continue
                    list_board_indices.extend([(i, j, k) for k in range(len(list(game.mainline_moves())))])

        return list_board_indices

    @logger.catch
    def retrieve_board(self, idx: int) -> (chess.Board, int, int, str):
        """Retrieve the board at the given index of the dataset.

        Args:
            idx (int): Index of the board to retrieve.

        Returns:
            board (chess.Board): The board at the given index.
            move_id (int): The latest move index in the game.
            total_moves (int): The total number of moves in the game.
            result (str): The result of the game.
        """
        file_id, game_id, move_id = self.board_indices[idx]
        file = self.list_pgn_files[file_id]
        pgn = open(self.root_dir + "/" + file)

        for j in range(game_id):
            chess.pgn.skip_game(pgn)
        game = chess.pgn.read_game(pgn)

        result = game.headers["Result"]
        board = game.board()
        mainline = list(game.mainline_moves())
        for move in mainline[:move_id]:
            board.push(move)
        return board, move_id, len(mainline), result

    @logger.catch
    def __len__(self):
        return len(self.board_indices)

    @logger.catch
    def __getitem__(self, idx: Union[Tensor, int]):
        if torch.is_tensor(idx):
            idx = int(idx.item())

        board_sample, move_id, game_len, game_result = self.retrieve_board(idx)
        legal_moves_sample = list(board_sample.legal_moves)
        outcome = {"move_id": move_id,
                   "game_length": game_len,
                   "game_result": game_result}

        if self.transform:
            board_sample = torch.from_numpy(board_to_tensor(board_sample))
            legal_moves_sample = torch.from_numpy(moves_to_tensor(legal_moves_sample))
            outcome = torch.tensor([move_id,
                                    game_len,
                                    result_to_tensor(game_result)[0]])

        if self.return_moves and self.return_outcome:
            return board_sample, legal_moves_sample, outcome

        if self.return_moves:
            return board_sample, legal_moves_sample

        if self.return_outcome:
            return board_sample, outcome

        return board_sample

    @logger.catch
    def __getitems__(self, indices: Union[Tensor, list[int]]):
        if torch.is_tensor(indices):
            indices = indices.tolist()

        board_samples = []
        legal_moves_samples = []
        outcomes = []

        for i in indices:
            board_sample, move_id, game_len, game_result = self.retrieve_board(i)
            legal_moves_sample = list(board_sample.legal_moves)
            outcome = {"move_id": move_id,
                       "game_length": game_len,
                       "game_result": game_result}

            board_samples.append(board_sample)
            legal_moves_samples.append(legal_moves_sample)
            outcomes.append(outcome)

        if self.transform:
            logger.info("Transforming the boards to tensors...")
            board_samples = torch.from_numpy(batch_boards_to_tensor(board_samples))
            logger.info("Transforming the legal moves to tensors...")
            legal_moves_samples = torch.from_numpy(batch_moves_to_tensor(legal_moves_samples))
            moves_ids = np.array([outcome["move_id"] for outcome in outcomes])
            game_lens = np.array([outcome["game_length"] for outcome in outcomes])
            logger.info("Transforming the outcomes to tensors...")
            game_results = batch_results_to_tensor([outcome["game_result"] for outcome in outcomes]).flatten()
            outcomes = torch.tensor(np.array([moves_ids,
                                              game_lens,
                                              game_results]).T)

        if self.return_moves and self.return_outcome:
            return board_samples, legal_moves_samples, outcomes

        if self.return_moves:
            return board_samples, legal_moves_samples

        if self.return_outcome:
            return board_samples, outcomes

        return board_samples
