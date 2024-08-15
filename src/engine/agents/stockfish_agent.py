import chess.pgn
from stockfish import Stockfish

from src.engine.agents.base_agent import BaseAgent


class StockfishAgent(BaseAgent):
    def __init__(self, is_white: bool, elo: int = 1300, depth: int = 5, n_threads: int = 1,
                 memory_hash: int = 2048) -> None:
        super().__init__(is_white)
        self.depth = depth
        self.elo = elo
        self.n_threads = n_threads
        self.memory_hash = memory_hash

    def get_stockfish(self) -> Stockfish:
        return Stockfish(
            depth=self.depth,
            parameters=
            {
                "Debug Log File": "",
                "Contempt": 0,
                "Min Split Depth": 0,
                "Threads": self.n_threads,  # The number of CPU threads used for searching a position.
                "Ponder": "false",  # Let Stockfish ponder its next move while the opponent is thinking.
                "Hash": self.memory_hash,  # The size of the hash table in MB.
                "MultiPV": 1,  # Output the N best lines (principal variations, PVs) when searching.
                "Skill Level": 20,
                # Lower the Skill Level in order to make Stockfish play weaker (see also UCI_LimitStrength).
                "Move Overhead": 10,  # Assume a time delay of x ms due to network and GUI overheads.
                "Minimum Thinking Time": 20,
                "Slow Mover": 100,
                "UCI_Chess960": "false",
                "UCI_LimitStrength": "true",  # Enable weaker play aiming for an Elo rating as set by UCI_Elo
                "UCI_Elo": self.elo  # If UCI_LimitStrength is enabled, it aims for an engine strength of the given Elo.
            }
        )

    def next_move(self, board: chess.Board) -> chess.Move:
        stockfish = self.get_stockfish()
        stockfish.set_fen_position(board.fen())
        move = stockfish.get_best_move()
        return chess.Move.from_uci(move)

    def evaluate_board(self, board: chess.Board) -> float:
        stockfish = self.get_stockfish()
        stockfish.set_fen_position(board.fen())
        board_eval = stockfish.get_evaluation()
        if board_eval["type"] == "cp":
            return board_eval["value"] / 100
        else:
            return -100 if board_eval["value"] < 0 else 100

    def __str__(self) -> str:
        return "StockfishAgent"
