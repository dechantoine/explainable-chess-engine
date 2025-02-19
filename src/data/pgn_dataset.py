import hashlib
import os
from typing import Union

import chess.pgn
import numpy as np
import torch
from loguru import logger
from pympler import asizeof
from torch import Tensor
from tqdm.contrib.concurrent import process_map

from src.data.base_dataset import BaseChessDataset, BoardItem
from src.data.data_utils import batch_boards_to_tensor, board_to_tensor, result_to_tensor


class PGNDataset(BaseChessDataset):
    """Chess boards with legal moves dataset."""

    @logger.catch
    def __init__(
            self,
            root_dir: str,
            include_draws: bool = False,
            in_memory: bool = False,
            num_workers: int = 1,
            winner: bool = False,
            move_count: bool = False
    ):
        """
        Arguments:
            root_dir (string): Directory with all the PGNs.
            include_draws (bool): Include draws in the dataset.
            in_memory (bool): Load the dataset in memory. Use with caution for large datasets. Returns transformed tensors.
            num_workers (int): Number of workers for the DataLoader. Only used if in_memory is True.
            winner (bool): Whether to include winner in outputs.
            move_count (bool): Whether to include move count in outputs.
        """
        super().__init__(stockfish_eval=False,
                         winner=winner,
                         move_count=move_count)

        self.root_dir = root_dir
        self.list_pgn_files = [
            f for f in os.listdir(self.root_dir) if f.endswith(".pgn")
        ]
        self.list_pgn_files.sort()
        self.board_indices = self.get_boards_indices(
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
    ) -> list[tuple[int, int, int]]:
        """Get the indices of all the boards in the dataset.

        Args:
            include_draws (bool): Include draws in the dataset.

        Returns: list[tuple[int, int, int]] and list[int]: List of tuples containing the file index, game index,
        and move index

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

                    if len(list(game.mainline_moves())) == 1:
                        logger.warning(
                            f"Game {j} in file {file} has one move, skipping."
                        )
                        continue

                    list_board_indices.extend(
                        [(i, j, k) for k in range(1, len(list(game.mainline_moves()))+1)]
                    )

        return list_board_indices

    @logger.catch
    def retrieve_board(self, idx: int) -> (chess.Board, int, int, int):
        """Retrieve the board at the given index of the dataset from files.

        Args:
            idx (int): Index of the board to retrieve.

        Returns:
            board (chess.Board): The board at the given index.
            move_id (int): The latest move index in the game.
            total_moves (int): The total number of moves in the game.
            result (int): The result of the game.

        """
        file_id, game_id, move_id = self.board_indices[idx]
        file = self.list_pgn_files[file_id]
        pgn = open(self.root_dir + "/" + file)

        for j in range(game_id):
            chess.pgn.skip_game(pgn)
        game = chess.pgn.read_game(pgn)

        result = int(result_to_tensor(game.headers["Result"])[0])
        board = game.board()
        mainline = list(game.mainline_moves())
        for move in mainline[:move_id]:
            board.push(move)
        return board, (move_id // 2) + 1, (len(mainline) // 2) + 1, result

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

        board_samples = list(board_samples)
        self.board_samples = [board.fen() for board in board_samples]

        if self.winner:
            self.winners = game_results

        if self.move_count:
            self.moves_ids = move_ids
            self.total_moves = game_lens

    @logger.catch
    def __len__(self) -> int:
        return len(self.board_indices)

    @logger.catch
    def __getitem__(self, idx: Union[Tensor, int]) -> BoardItem:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        if self.in_memory:
            board_sample = self.board_samples[idx]
            board_sample = chess.Board(fen=board_sample)

            winner = self.winners[idx] if self.winner else None
            move_id = self.moves_ids[idx] if self.move_count else None
            total_moves = self.total_moves[idx] if self.move_count else None

        else:
            board_sample, move_id, total_moves, winner = self.retrieve_board(idx=idx)

        board_array, active_color, castling = board_to_tensor(board=board_sample)

        board_sample = torch.tensor(board_array)
        active_color = torch.tensor(active_color)
        castling = torch.tensor(castling)

        if self.winner:
            winner = torch.tensor([winner])

        if self.move_count:
            move_id = torch.tensor([move_id])
            total_moves = torch.tensor([total_moves])

        return BoardItem(
            board=board_sample,
            active_color=active_color,
            castling=castling,
            winner=winner if self.winner else None,
            move_id=move_id if self.move_count else None,
            total_moves=total_moves if self.move_count else None
        )

    @logger.catch
    def __getitems__(self, indices: Union[Tensor, list[int]]):
        if torch.is_tensor(indices):
            indices = indices.int().tolist()

        if self.in_memory:
            board_samples = [self.board_samples[i] for i in indices]
            board_samples = [chess.Board(fen=board_sample) for board_sample in board_samples]

            winners = [self.winners[i] for i in indices] if self.winner else None
            move_ids = [self.moves_ids[i] for i in indices] if self.move_count else None
            totals_moves = [self.total_moves[i] for i in indices] if self.move_count else None

        else:
            board_samples, move_ids, totals_moves, winners = zip(
                *[self.retrieve_board(idx=i) for i in indices]
            )

        board_samples, active_colors, castlings = batch_boards_to_tensor(
            batch_boards=board_samples
        )

        if self.winner:
            winners = torch.tensor([[w] for w in winners])

        if self.move_count:
            move_ids = torch.tensor([[m] for m in move_ids])
            totals_moves = torch.tensor([[t] for t in totals_moves])

        return BoardItem(
            board=board_samples,
            active_color=active_colors,
            castling=castlings,
            winner=winners if self.winner else None,
            move_id=move_ids if self.move_count else None,
            total_moves=totals_moves if self.move_count else None
        )

    @logger.catch
    def log_memory(self):
        """Log the memory usage of the dataset."""
        memusage = {
            "board_samples": asizeof.asizeof(self.board_samples) / 1e6,
        }
        if self.move_count:
            memusage["moves_id"] = (
                    asizeof.asizeof(self.moves_ids) / 1e6
            )
            memusage["total_moves"] = (
                    asizeof.asizeof(self.total_moves) / 1e6
            )
        if self.winner:
            memusage["winners"] = asizeof.asizeof(self.winners) / 1e6

        report = "Dataset loaded in memory. Memory usage: "
        for key, value in memusage.items():
            report += f"{key}: {value:.2f} MB, "
        logger.info(report)
