import os
from typing import Union

import chess.pgn
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import board_to_tensor, moves_to_tensor, batch_boards_to_tensor, batch_moves_to_tensor
from loguru import logger


class ChessBoardDataset(Dataset):
    """Chess boards with legal moves dataset."""

    @logger.catch
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the PGNs.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.list_pgn_files = [f for f in os.listdir(self.root_dir) if f.endswith(".pgn")]
        self.board_indices = self.get_boards_indices()

    @logger.catch
    def get_boards_indices(self):
        list_board_indices = []
        for i, file in enumerate(self.list_pgn_files):
            pgn = open(self.root_dir + "/" + file)
            j = 0
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                list_board_indices.extend([(i, j, k) for k in range(len(list(game.mainline_moves())))])
                j += 1
        return list_board_indices


    @logger.catch
    def retrieve_board(self, idx: int):
        file_id, game_id, move_id = self.board_indices[idx]
        file = self.list_pgn_files[file_id]
        logger.info(f"file: {file}, game_id: {game_id}, move_id: {move_id}")
        pgn = open(self.root_dir + "/" + file)

        game = chess.pgn.read_game(pgn)
        for j in range(game_id):
            game = chess.pgn.read_game(pgn)

        board = game.board()
        for move in list(game.mainline_moves())[:move_id]:
            board.push(move)
        return board

    @logger.catch
    def __len__(self):
        return len(self.board_indices)

    @logger.catch
    def __getitem__(self, idx: Union[Tensor, int]):
        if torch.is_tensor(idx):
            idx = int(idx.item())

        board_sample = self.retrieve_board(idx)
        legal_moves_sample = list(board_sample.legal_moves)

        if self.transform:
            board_sample = torch.tensor(board_to_tensor(board_sample))

        if self.target_transform:
            legal_moves_sample = torch.tensor(moves_to_tensor(legal_moves_sample))

        return board_sample, legal_moves_sample

    @logger.catch
    def __getitems__(self, indices: Union[Tensor, list[int]]):
        if torch.is_tensor(indices):
            indices = indices.tolist()

        board_samples = []
        legal_moves_samples = []

        for i in indices:
            board = self.retrieve_board(i)
            board_samples.append(board)
            legal_moves_samples.append(list(board.legal_moves))

        if self.transform:
            board_samples = torch.tensor(batch_boards_to_tensor(board_samples))

        if self.target_transform:
            legal_moves_samples = torch.tensor(batch_moves_to_tensor(legal_moves_samples))

        return board_samples, legal_moves_samples
