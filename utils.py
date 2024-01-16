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


@logger.catch
def format_board(board: chess.Board) -> str:
    return str(board).replace("\n", " ").replace(" ", "")


# TODO: read from FEN
@logger.catch
def string_to_array(str_board: str, is_white: bool = True) -> np.array:
    list_board = list(str_board)
    key = "white" if is_white else "black"
    return np.array([np.reshape([1*(p == piece) for p in list_board],
                                newshape=(8, 8))
                     for piece in list(dict_pieces[key])])


@logger.catch
def game_to_tensor(game: chess.pgn.Game) -> np.array:
    board = game.board()
    boards_tensors = []
    for move in game.mainline_moves():
        board.push(move)
        str_board = format_board(board)
        boards_tensors.append([string_to_array(str_board),
                               string_to_array(str_board, is_white=False)])
    return np.array(boards_tensors)


@logger.catch
def batch_game_to_tensor(batch_games: list) -> list:
    return [game_to_tensor(game) for game in tqdm(batch_games)]
