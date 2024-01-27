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


#TODO: test for castling !
@logger.catch
def uci_to_coordinates(move: chess.Move) -> tuple:
    return (7 - move.from_square // 8, move.from_square % 8),\
        (7 - move.to_square // 8, move.to_square % 8)


@logger.catch
def board_to_legal_moves_tensor(board: chess.Board) -> np.array:
    legal_moves = board.legal_moves
    moves_tensor = np.zeros(shape=(8*8, 8*8))
    for move in legal_moves:
        from_coordinates, to_coordinates = uci_to_coordinates(move)
        moves_tensor[from_coordinates[0]*8 + from_coordinates[1],
                     to_coordinates[0]*8 + to_coordinates[1]] = 1
    return moves_tensor


@logger.catch
def batch_boards_to_legal_moves_tensor(batch_boards: list[chess.Board]) -> list[np.array]:
    return [board_to_legal_moves_tensor(board) for board in tqdm(batch_boards)]


@logger.catch
def batch_boards_to_board_tensor(batch_boards: list[chess.Board]) -> list[np.array]:
    return [string_to_array(format_board(board)) for board in tqdm(batch_boards)]


@logger.catch
def game_to_legal_moves_tensor(game: chess.pgn.Game) -> np.array:
    board = game.board()
    boards = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    legal_moves_tensors = batch_boards_to_legal_moves_tensor(boards)
    return np.array(legal_moves_tensors)


@logger.catch
def batch_game_to_legal_moves_tensor(batch_games: list[chess.pgn.Game]) -> list[np.array]:
    return [game_to_legal_moves_tensor(game) for game in tqdm(batch_games)]


@logger.catch
def game_to_board_tensor(game: chess.pgn.Game) -> np.array:
    board = game.board()
    boards = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    board_tensors = batch_boards_to_board_tensor(boards)
    return np.array(board_tensors)


@logger.catch
def batch_game_to_board_tensor(batch_games: list[chess.pgn.Game]) -> list[np.array]:
    return [game_to_board_tensor(game) for game in tqdm(batch_games)]

