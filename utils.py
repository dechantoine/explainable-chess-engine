import numpy as np
from loguru import logger
from tqdm import tqdm
import chess.pgn

dict_pieces = {"white": {"R": "rook",
                         "N": "knight",
                         "B": "bishop",
                         "Q": "queen",
                         "K": "king",
                         "P": "pawn"},
               "black": {"r": "rook",
                         "n": "knight",
                         "b": "bishop",
                         "q": "queen",
                         "k": "king",
                         "p": "pawn"}}


# TODO: read from FEN
@logger.catch
def string_to_array(str_board: str, is_white: bool = True):
    list_board = list(str_board.replace("\n", "").replace(" ", ""))
    key = "white" if is_white else "black"
    return np.array([np.reshape([1 if p == piece else 0 for p in list_board],
                                newshape=(8, 8))
                     for piece in list(dict_pieces[key])])


@logger.catch
def game_to_tensor(game: chess.pgn.Game):
    board = game.board()
    boards_tensors = []
    for move in game.mainline_moves():
        board.push(move)
        boards_tensors.append([string_to_array(str(board)),
                               string_to_array(str(board), is_white=False)])
    return np.array(boards_tensors)


@logger.catch
def batch_game_to_tensor(batch_games: list):
    return [game_to_tensor(game) for game in tqdm(batch_games)]