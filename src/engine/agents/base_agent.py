from abc import ABC

import chess.pgn


class BaseAgent(ABC):
    def __init__(self, is_white: bool) -> None:
        self.is_white = is_white

    def next_move(self, board: chess.Board) -> chess.Move:
        pass

    def evaluate_board(self, board: chess.Board) -> float:
        pass
