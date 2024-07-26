import chess.pgn
from stockfish import Stockfish

from src.engine.agents.base_agent import BaseAgent


class StockfishAgent(BaseAgent):
    def __init__(self, is_white: bool, elo: int = 1300, depth: int = 5, n_threads: int = 1,
                 memory_hash: int = 2048) -> None:
        super().__init__(is_white)
        self.stockfish = Stockfish(
            depth=depth,
            parameters=
            {
                "Debug Log File": "",
                "Contempt": 0,
                "Min Split Depth": 0,
                "Threads": n_threads,  # The number of CPU threads used for searching a position.
                "Ponder": "false",  # Let Stockfish ponder its next move while the opponent is thinking.
                "Hash": memory_hash,  # The size of the hash table in MB.
                "MultiPV": 1,  # Output the N best lines (principal variations, PVs) when searching.
                "Skill Level": 20,  # Lower the Skill Level in order to make Stockfish play weaker (see also UCI_LimitStrength).
                "Move Overhead": 10,  # Assume a time delay of x ms due to network and GUI overheads.
                "Minimum Thinking Time": 20,
                "Slow Mover": 100,
                "UCI_Chess960": "false",
                "UCI_LimitStrength": "false",  # Enable weaker play aiming for an Elo rating as set by UCI_Elo
                "UCI_Elo": elo  # If UCI_LimitStrength is enabled, it aims for an engine strength of the given Elo.
            }
        )

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

    def __str__(self) -> str:
        return "StockfishAgent"
