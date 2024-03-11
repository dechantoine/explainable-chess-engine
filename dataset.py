import os
from typing import Union

import chess.pgn
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import format_board
from loguru import logger


class ChessBoardDataset(Dataset):
    """Chess boards with legal moves dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the PGNs.
        """
        self.root_dir = root_dir
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

    def __len__(self):
        return len(self.board_indices)

    def __getitem__(self, indices: Union[Tensor, list[int]]):
        if torch.is_tensor(indices):
            indices = indices.tolist()

        sample = dict()

        for i in indices:

            file_id, game_id, move_id = self.board_indices[i]
            file = self.list_pgn_files[file_id]
            logger.info(f"file: {file}, game_id: {game_id}, move_id: {move_id}")
            pgn = open(self.root_dir + "/" + file)

            game = chess.pgn.read_game(pgn)
            for j in range(game_id):
                game = chess.pgn.read_game(pgn)

            board = game.board()
            for move in list(game.mainline_moves())[:move_id]:
                board.push(move)

            sample[i] = {"board": format_board(board),
                         "legal_moves": list(board.legal_moves)}

        #if self.transform:
        #    sample = self.transform(sample)

        return sample
