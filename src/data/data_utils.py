import io
import re

import chess.pgn
import numpy as np
import torch
from loguru import logger

dict_pieces = {
    "white": {
        "R": "rook",
        "N": "knight",
        "B": "bishop",
        "Q": "queen",
        "K": "king",
        "P": "pawn",
    },
    "black": {
        "r": "rook",
        "n": "knight",
        "b": "bishop",
        "q": "queen",
        "k": "king",
        "p": "pawn",
    },
}


def arrays_to_lists(data):
    """Recursively transform all numpy arrays in a nested structure into lists.

    Args:
        data: The nested structure containing numpy arrays.

    Returns:
        The nested structure with all numpy arrays converted to lists.

    """
    if isinstance(data, np.ndarray):
        data = data.tolist()
        return [arrays_to_lists(item) for item in data]
    elif isinstance(data, list):
        return [arrays_to_lists(item) for item in data]
    else:
        return data


@logger.catch
def clean_board(board: str) -> chess.Board:
    """Clean the board string and return a chess.Board object.

    Args:
        board (str): board string

    Returns:
        chess.Board: chess.Board object

    """
    board = board.replace("'", "")
    board = board.replace('"', "")

    try:
        board = chess.Board(fen=board)
    except ValueError:
        try:
            game = chess.pgn.read_game(io.StringIO(board))
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
        except ValueError:
            raise ValueError("Invalid FEN or PGN board provided.")

    return board


@logger.catch
def format_board(board: chess.Board) -> str:
    """Format a board to a compact string.

    Args:
        board (chess.Board): board to format.

    Returns:
        str: formatted board.

    """
    return str(board).replace("\n", "").replace(" ", "")


@logger.catch
def string_to_array(str_board: str, is_white: bool = True) -> np.array:
    """Convert a string compact board to a numpy array. The array is of shape (6, 8, 8) and is the one-hot encoding of
    the player pieces.

    Args:
        str_board (str): compact board.
        is_white (bool, optional): True if white pieces, False otherwise. Defaults to True.

    Returns:
        np.array: numpy array of shape (6, 8, 8).

    """
    list_board = list(str_board)
    key = "white" if is_white else "black"
    return np.array(
        [
            np.reshape([1 * (p == piece) for p in list_board], newshape=(8, 8))
            for piece in list(dict_pieces[key])
        ]
    )


def board_to_list_index(board: chess.Board) -> list:
    """Convert a chess board to a list of indexes.

    Args:
        board (chess.Board): board to convert.

    Returns:
        list: list of indexes.

    """
    list_board = list(format_board(board))
    idx_white = [np.flatnonzero([1 * (p == piece) for p in list_board]).tolist()
                 for piece in list(dict_pieces["white"])]
    idx_black = [np.flatnonzero([1 * (p == piece) for p in list_board]).tolist()
                 for piece in list(dict_pieces["black"])]

    idx_white = [idx if len(idx) > 0 else None for idx in idx_white]
    idx_black = [idx if len(idx) > 0 else None for idx in idx_black]

    active_color = 1 * (board.turn == chess.WHITE)

    castling = [board.has_kingside_castling_rights(chess.WHITE) * 1,
                board.has_queenside_castling_rights(chess.WHITE) * 1,
                board.has_kingside_castling_rights(chess.BLACK) * 1,
                board.has_queenside_castling_rights(chess.BLACK) * 1]

    en_passant = board.ep_square if board.ep_square else -1

    list_indexes = idx_white + idx_black + [active_color] + [castling] + [en_passant] + [board.halfmove_clock] + [
        board.fullmove_number]

    return list_indexes


def list_index_to_fen(idxs: list) -> str:
    """Convert a list of indexes to a FEN string.

    Args:
        idxs (list): list of indexes.

    Returns:
        str: FEN string.

    """
    idx_white = idxs[:6]
    idx_black = idxs[6:12]
    active_color, castling, en_passant, halfmove, fullmove = idxs[12:]
    list_board = ["."] * 64
    for i, piece in enumerate(list(dict_pieces["white"])):
        if idx_white[i]:
            for idx in idx_white[i]:
                list_board[idx] = piece
    for i, piece in enumerate(list(dict_pieces["black"])):
        if idx_black[i]:
            for idx in idx_black[i]:
                list_board[idx] = piece
    for k in range(7):
        list_board.insert(8 * (k + 1) + k, "/")

    active_color = "w" if active_color else "b"

    str_castling = ["K" if castling[0] else "",
                    "Q" if castling[1] else "",
                    "k" if castling[2] else "",
                    "q" if castling[3] else ""]
    str_castling = "".join(str_castling)
    str_castling = str_castling if str_castling else "-"

    en_passant = chess.SQUARE_NAMES[en_passant] if en_passant != -1 else "-"

    fen = ("".join(list_board) + " "
           + active_color + " "
           + str_castling + " "
           + str(en_passant) + " "
           + str(halfmove) + " "
           + str(fullmove))
    fen = re.sub(r'\.+', lambda m: str(len(m.group())), fen)
    return fen


def list_index_to_tensor(idxs: list) -> np.array:
    """Convert a list of indexes to a tensor.

    Args:
        idxs (list): list of indexes.

    Returns:
        np.array: tensor.

    """
    tensor_pieces = np.zeros((12, 8 * 8), dtype=np.int8)
    for i, list_idx in enumerate(idxs[:12]):
        if list_idx:
            for idx in list_idx:
                tensor_pieces[i, idx] = 1
    tensor_pieces = tensor_pieces.reshape((12, 8, 8))

    return tensor_pieces


@logger.catch
def uci_to_coordinates(move: chess.Move) -> tuple:
    """Convert a move in UCI format to coordinates.

    Args:
        move (chess.Move): move to convert.

    Returns:
        tuple: coordinates of the origin square and coordinates of the destination square.

    """
    return (7 - move.from_square // 8, move.from_square % 8), (
        7 - move.to_square // 8,
        move.to_square % 8,
    )


@logger.catch
def moves_to_tensor(moves: list[chess.Move]) -> np.array:
    """Convert a list of moves to a (8*8, 8*8) tensor. For each origin square, the tensor contains a vector of size 8*8
    with 1 at the index of the destination squares in list of moves, 0 otherwise.

    Args:
        moves (list[chess.Move]): list of moves.

    Returns:
        np.array: tensor of possible moves from each square.

    """
    moves_tensor = np.zeros(shape=(8 * 8, 8 * 8), dtype=np.int8)
    for move in moves:
        from_coordinates, to_coordinates = uci_to_coordinates(move)
        moves_tensor[
            from_coordinates[0] * 8 + from_coordinates[1],
            to_coordinates[0] * 8 + to_coordinates[1],
        ] = 1
    return moves_tensor


@logger.catch
def board_to_tensor(board: chess.Board) -> tuple[np.array, np.array, np.array]:
    """Convert a board to a tuple of shapes ((12, 8, 8), (1) , (4)). The tuple contains the one-hot encoding of the
    board, the active color and the castling rights.

    Args:
        board (chess.Board): board to convert.

    Returns:
        tuple[np.array, np.array, np.array]: tuple of tensors.

    """
    list_board = list(format_board(board))

    idx_white = [np.flatnonzero([1 * (p == piece) for p in list_board]).tolist()
                 for piece in list(dict_pieces["white"])]
    idx_black = [np.flatnonzero([1 * (p == piece) for p in list_board]).tolist()
                 for piece in list(dict_pieces["black"])]

    active_color = 1 * (board.turn == chess.WHITE)

    castling = [board.has_kingside_castling_rights(chess.WHITE) * 1,
                board.has_queenside_castling_rights(chess.WHITE) * 1,
                board.has_kingside_castling_rights(chess.BLACK) * 1,
                board.has_queenside_castling_rights(chess.BLACK) * 1]

    return list_index_to_tensor(idx_white + idx_black), np.array([active_color]), np.array(castling)


@logger.catch
def batch_moves_to_tensor(batch_moves: list[list[chess.Move]]) -> np.array:
    """Convert a batch of list of moves to a batch of (8*8, 8*8) tensors.

    Args:
        batch_moves (list[list[chess.Move]]): batch of list of moves.

    Returns:
        list[np.array]: batch of moves tensors.

    """

    return np.array([moves_to_tensor(moves) for moves in batch_moves])


@logger.catch
def batch_boards_to_tensor(
        batch_boards: list[chess.Board]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a batch of boards to a batch of board tensors.

    Args:
        batch_boards (list[chess.Board]): batch of boards to convert.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of tensors.

    """
    tensors = [board_to_tensor(board) for board in batch_boards]
    return (torch.tensor(np.array([tensors[i][0] for i in range(len(tensors))])),
            torch.tensor(np.array([tensors[i][1] for i in range(len(tensors))])),
            torch.tensor(np.array([tensors[i][2] for i in range(len(tensors))])))


@logger.catch
def game_to_legal_moves_tensor(game: chess.pgn.Game) -> np.array:
    """Convert a game to a tensor of legal moves. The tensor is of shape (nb_moves, 8*8, 8*8) and contains a tensor of
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
    legal_moves_tensors = batch_moves_to_tensor(
        [list(board.legal_moves) for board in boards]
    )
    return np.array(legal_moves_tensors)


@logger.catch
def game_to_board_tensor(game: chess.pgn.Game) -> np.array:
    """Convert a game to a tensor of boards. The tensor is of shape (nb_moves, 12, 8, 8) and contains a board tensor for
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
    board_tensors = batch_boards_to_tensor(boards)
    return np.array(board_tensors)


@logger.catch(exclude=ValueError)
def result_to_tensor(result: str) -> np.array:
    """Convert a game result to a tensor. The tensor is of shape (1,) and contains 1 for a white win, 0 for a draw and
    -1 for a white loss.

    Args:
        result (str): game result.

    Returns:
        np.array: tensor of game result.

    """
    if result == "1-0":
        return np.array([1], dtype=np.int8)
    elif result == "0-1":
        return np.array([-1], dtype=np.int8)
    elif result == "1/2-1/2":
        return np.array([0], dtype=np.int8)
    else:
        raise ValueError(f"Result {result} not supported.")


@logger.catch
def batch_results_to_tensor(batch_results: list[str]) -> np.array:
    """Convert a batch of game results to a tensor. The tensor is of shape (nb_games, 1) and contains a tensor of game
    result for each game of the batch.

    Args:
        batch_results (list[str]): batch of game results.

    Returns:
        np.array: tensor of game results.

    """
    return np.array([result_to_tensor(result) for result in batch_results])


@logger.catch
def read_boards_from_pgn(pgn_file: str, start_move: int = 0, end_move: int = 0) -> list[chess.Board]:
    """Read boards from a PGN file.

    Args:
        pgn_file (str): path to the PGN file
        start_move (int): move to start from in each game
        end_move (int): move to end at in each game (counting from the end)

    Returns:
        list[chess.Board]: list of boards

    """
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)
    boards = []

    while game:
        board = game.board()
        mainline = list(game.mainline_moves())
        end_index = len(mainline) - end_move

        for i, move in enumerate(mainline[:end_index]):
            board.push(move)
            if start_move <= i:
                boards.append(board.copy())
        game = chess.pgn.read_game(pgn)

    return boards
