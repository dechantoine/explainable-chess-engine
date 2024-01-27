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
    """
    Format a board to a compact string.

    Args:
        board (chess.Board): board to format.

    Returns:
        str: formatted board.
    """
    return str(board).replace("\n", " ").replace(" ", "")


# TODO: read from FEN
@logger.catch
def string_to_array(str_board: str, is_white: bool = True) -> np.array:
    """
    Convert a string compact board to a numpy array. The array is of shape (6, 8, 8) and is the one-hot encoding of
    the player pieces.

    Args:
        str_board (str): compact board.
        is_white (bool, optional): True if white pieces, False otherwise. Defaults to True.

    Returns:
        np.array: numpy array of shape (6, 8, 8).
    """
    list_board = list(str_board)
    key = "white" if is_white else "black"
    return np.array([np.reshape([1*(p == piece) for p in list_board],
                                newshape=(8, 8))
                     for piece in list(dict_pieces[key])])


#TODO: test for castling !
@logger.catch
def uci_to_coordinates(move: chess.Move) -> tuple:
    """
    Convert a move in UCI format to coordinates.

    Args:
        move (chess.Move): move to convert.

    Returns:
        tuple: coordinates of the origin square and coordinates of the destination square.
    """
    return (7 - move.from_square // 8, move.from_square % 8),\
        (7 - move.to_square // 8, move.to_square % 8)


@logger.catch
def board_to_legal_moves_tensor(board: chess.Board) -> np.array:
    """
    Convert a board to a (8*8, 8*8) tensor of legal moves. For each origin square, the tensor contains a vector of size
    8*8 with 1 at the index of the destination square if the move is legal, 0 otherwise.

    Args:
        board (chess.Board): board to convert.

    Returns:
        np.array: tensor of legal moves
    """
    legal_moves = board.legal_moves
    moves_tensor = np.zeros(shape=(8*8, 8*8))
    for move in legal_moves:
        from_coordinates, to_coordinates = uci_to_coordinates(move)
        moves_tensor[from_coordinates[0]*8 + from_coordinates[1],
                     to_coordinates[0]*8 + to_coordinates[1]] = 1
    return moves_tensor


@logger.catch
def batch_boards_to_legal_moves_tensor(batch_boards: list[chess.Board]) -> list[np.array]:
    """
    Convert a batch of boards to a batch of legal moves tensors.

    Args:
        batch_boards (list[chess.Board]): batch of boards to convert.

    Returns:
        list[np.array]: batch of legal moves tensors.
    """

    return [board_to_legal_moves_tensor(board) for board in tqdm(batch_boards)]


@logger.catch
def batch_boards_to_board_tensor(batch_boards: list[chess.Board]) -> list[np.array]:
    """
    Convert a batch of boards to a batch of board tensors.

    Args:
        batch_boards (list[chess.Board]): batch of boards to convert.

    Returns:
        list[np.array]: batch of board tensors.
    """
    return [np.concatenate((string_to_array(format_board(board)),
                            string_to_array(format_board(board), is_white=False)),
                           axis=0)
            for board in tqdm(batch_boards)]


@logger.catch
def game_to_legal_moves_tensor(game: chess.pgn.Game) -> np.array:
    """
    Convert a game to a tensor of legal moves. The tensor is of shape (nb_moves, 8*8, 8*8) and contains a tensor of
    legal moves for each move of the game.

    Args:
        game (chess.pgn.Game): game to convert.

    Returns:
        np.array: tensor of legal moves.
    """
    board = game.board()
    boards = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    legal_moves_tensors = batch_boards_to_legal_moves_tensor(boards)
    return np.array(legal_moves_tensors)


@logger.catch
def game_to_board_tensor(game: chess.pgn.Game) -> np.array:
    """
    Convert a game to a tensor of boards. The tensor is of shape (nb_moves, 12, 8, 8) and contains a board tensor for
    each move of the game.

    Args:
        game (chess.pgn.Game): game to convert.

    Returns:
        np.array: tensor of boards.
    """
    board = game.board()
    boards = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    board_tensors = batch_boards_to_board_tensor(boards)
    return np.array(board_tensors)

