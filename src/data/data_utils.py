import chess.pgn
import numpy as np
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


@logger.catch
def format_board(board: chess.Board) -> str:
    """Format a board to a compact string.

    Args:
        board (chess.Board): board to format.

    Returns:
        str: formatted board.

    """
    return str(board).replace("\n", "").replace(" ", "")


# TODO: read from FEN
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


# TODO: test for castling !
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
def board_to_tensor(board: chess.Board) -> np.array:
    """Convert a board to a (12, 8, 8) tensor. The tensor is the one-hot encoding of the board.

    Args:
        board (chess.Board): board to convert.

    Returns:
        np.array: board tensor.

    """
    return np.concatenate(
        (
            string_to_array(format_board(board)),
            string_to_array(format_board(board), is_white=False),
        ),
        axis=0,
        dtype=np.int8,
    )


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
def batch_boards_to_tensor(batch_boards: list[chess.Board]) -> np.array:
    """Convert a batch of boards to a batch of board tensors.

    Args:
        batch_boards (list[chess.Board]): batch of boards to convert.

    Returns:
        list[np.array]: batch of board tensors.

    """
    return np.array([board_to_tensor(board) for board in batch_boards])


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
