import chess.pgn
from stockfish import Stockfish

from src.agents.base_agent import BaseAgent


class StockfishAgent(BaseAgent):
    def __init__(self, is_white: bool, beam_width=5, beam_depth=11) -> None:
        super().__init__(is_white)
        self.stockfish = Stockfish(parameters=
        {
            "Debug Log File": "",
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 1,
            "Ponder": "false",
            "Hash": 16,
            "MultiPV": 1,
            "Skill Level": 20,
            "Move Overhead": 10,
            "Minimum Thinking Time": 20,
            "Slow Mover": 100,
            "UCI_Chess960": "false",
            "UCI_LimitStrength": "false",
            "UCI_Elo": 1350}
        )
        self.beam_width = beam_width
        self.beam_depth = beam_depth

    def next_move(self, board: chess.Board) -> chess.Move:
        self.stockfish.set_fen_position(board.fen())
        move = self.stockfish.get_best_move()
        return chess.Move.from_uci(move)

    def evaluate_board(self, board: chess.Board) -> float:
        self.stockfish.set_fen_position(board.fen())
        board_eval = self.stockfish.get_evaluation()
        if board_eval["type"] == "cp":
            return board_eval["value"] / 100
        else:
            return -1 if board_eval["value"] < 0 else 1
