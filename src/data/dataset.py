import hashlib
import os
from typing import Union

import chess.pgn
import numpy as np
import torch
from loguru import logger
from pympler import asizeof
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

from src.data.data_utils import (
    batch_boards_to_tensor,
    batch_moves_to_tensor,
    batch_results_to_tensor,
    board_to_tensor,
    moves_to_tensor,
    result_to_tensor,
)


class ChessBoardDataset(Dataset):
    """Chess boards with legal moves dataset."""

    @logger.catch
    def __init__(
        self,
        root_dir: str,
        return_moves: bool = False,
        return_outcome: bool = False,
        transform: bool = False,
        include_draws: bool = False,
        in_memory: bool = False,
        num_workers: int = None,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the PGNs.
            return_moves (bool): Return the legal moves for each board.
            return_outcome (bool): Return the outcome of the game for each board.
            transform (bool): Apply the transform to the boards and legal moves.
            include_draws (bool): Include draws in the dataset.
            in_memory (bool): Load the dataset in memory. Use with caution for large datasets.
            num_workers (int): Number of workers for the DataLoader. Only used if in_memory is True.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_moves = return_moves
        self.return_outcome = return_outcome
        self.list_pgn_files = [
            f for f in os.listdir(self.root_dir) if f.endswith(".pgn")
        ]
        self.board_indices, self.results = self.get_boards_indices(
            include_draws=include_draws
        )
        self.hash = self.get_hash()
        self.in_memory = in_memory
        self.num_workers = num_workers

        if self.in_memory:
            logger.info("Loading the dataset in memory...")
            self.load_in_memory()
            self.log_memory()

    @logger.catch
    def get_hash(self) -> str:
        """Get the hash of the dataset."""
        return hashlib.md5(
            (str(self.list_pgn_files) + str(self.board_indices)).encode()
        ).hexdigest()

    @logger.catch
    def get_boards_indices(
        self,
        include_draws: bool = False,
    ) -> list[tuple[int, int, int]] and list[int]:
        """Get the indices of all the boards in the dataset.

        Args:
            include_draws (bool): Include draws in the dataset.

        Returns: list[tuple[int, int, int]] and list[int]: List of tuples containing the file index, game index,
        and move index + List of results.

        """
        list_board_indices = []
        list_results = []
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

                    if len(list(game.mainline_moves())) == 1:
                        logger.warning(
                            f"Game {j} in file {file} has one move, skipping."
                        )
                        continue

                    list_board_indices.extend(
                        [(i, j, k) for k in range(len(list(game.mainline_moves())))]
                    )
                    list_results.extend(
                        [result[0] for _ in range(len(list(game.mainline_moves())))]
                    )

        return list_board_indices, list_results

    @logger.catch
    def retrieve_board(self, idx: int) -> (chess.Board, int, int, str):
        """Retrieve the board at the given index of the dataset from files.

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
    def load_in_memory(self):
        """Load the dataset in memory with multiprocessing."""
        retrieves = process_map(
            self.retrieve_board,
            range(self.__len__()),
            max_workers=self.num_workers,
            chunksize=self.__len__() // (100 * self.num_workers),
            desc=f"Retrieving boards with {self.num_workers} workers...",
        )
        board_samples, move_ids, game_lens, game_results = zip(*retrieves)

        self.board_samples = list(board_samples)
        self.legal_moves_samples = [
            list(board.legal_moves) for board in self.board_samples
        ]
        self.outcomes = [
            {
                "move_id": move_id,
                "game_length": game_len,
                "game_result": game_result,
            }
            for move_id, game_len, game_result in zip(move_ids, game_lens, game_results)
        ]

    @logger.catch
    def __len__(self):
        return len(self.board_indices)

    @logger.catch
    def __getitem__(self, idx: Union[Tensor, int]):
        if torch.is_tensor(idx):
            idx = int(idx.item())

        if self.in_memory:
            board_sample = self.board_samples[idx]
            legal_moves_sample = self.legal_moves_samples[idx]
            outcome = self.outcomes[idx]

        else:
            board_sample, move_id, game_len, game_result = self.retrieve_board(idx)
            legal_moves_sample = list(board_sample.legal_moves)
            outcome = {
                "move_id": move_id,
                "game_length": game_len,
                "game_result": game_result,
            }

        if self.transform:
            board_sample = torch.from_numpy(board_to_tensor(board_sample))
            legal_moves_sample = torch.from_numpy(moves_to_tensor(legal_moves_sample))
            move_id, game_len, game_result = outcome.values()
            outcome = torch.tensor(
                [move_id, game_len, result_to_tensor(game_result)[0]]
            )

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
            indices = indices.int().tolist()

        if self.in_memory:
            board_samples = [self.board_samples[i] for i in indices]
            legal_moves_samples = [self.legal_moves_samples[i] for i in indices]
            outcomes = [self.outcomes[i] for i in indices]

        else:
            board_samples, move_ids, game_lens, game_results = zip(
                *[self.retrieve_board(i) for i in indices]
            )

            board_samples = list(board_samples)
            legal_moves_samples = [list(board.legal_moves) for board in board_samples]
            outcomes = [
                {
                    "move_id": move_id,
                    "game_length": game_len,
                    "game_result": game_result,
                }
                for move_id, game_len, game_result in zip(
                    move_ids, game_lens, game_results
                )
            ]

        if self.transform:
            # logger.info("Transforming the boards to tensors...")
            board_samples = torch.from_numpy(batch_boards_to_tensor(board_samples))
            # logger.info("Transforming the legal moves to tensors...")
            legal_moves_samples = torch.from_numpy(
                batch_moves_to_tensor(legal_moves_samples)
            )
            moves_ids = np.array([outcome["move_id"] for outcome in outcomes])
            game_lens = np.array([outcome["game_length"] for outcome in outcomes])
            # logger.info("Transforming the outcomes to tensors...")
            game_results = batch_results_to_tensor(
                [outcome["game_result"] for outcome in outcomes]
            ).flatten()
            outcomes = torch.tensor(np.array([moves_ids, game_lens, game_results]).T)

        if self.return_moves and self.return_outcome:
            return board_samples, legal_moves_samples, outcomes

        if self.return_moves:
            return board_samples, legal_moves_samples

        if self.return_outcome:
            return board_samples, outcomes

        return board_samples

    @logger.catch
    def log_memory(self):
        """Log the memory usage of the dataset."""
        memusage = {
            "board_indices": asizeof.asizeof(self.board_indices) / 1e6,
            "board_samples": asizeof.asizeof(self.board_samples) / 1e6,
        }
        if self.return_moves:
            memusage["legal_moves_samples"] = (
                asizeof.asizeof(self.legal_moves_samples) / 1e6
            )
        if self.return_outcome:
            memusage["outcomes"] = asizeof.asizeof(self.outcomes) / 1e6

        report = "Dataset loaded in memory. Memory usage: "
        for key, value in memusage.items():
            report += f"{key}: {value:.2f} MB, "
        logger.info(report)
