import io

import chess.pgn


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
