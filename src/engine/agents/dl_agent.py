import chess.pgn
import numpy as np
from anytree import LevelOrderGroupIter

from src.engine.agents.base_agent import BaseAgent
from src.engine.agents.policies import beam_search, eval_board


class DLAgent(BaseAgent):
    def __init__(self, model, is_white: bool, beam_width=5, beam_depth=11) -> None:
        super().__init__(is_white)
        self.model = model
        self.beam_width = beam_width
        self.beam_depth = beam_depth

    def next_move(self, board: chess.Board) -> chess.Move:
        beam = list(LevelOrderGroupIter(beam_search(self.model, board, self.beam_depth, self.beam_width)))
        if self.is_white:
            idx_best_node = np.argmax([node.score for node in beam[self.beam_depth - 1]])
        else:
            idx_best_node = np.argmin([node.score for node in beam[self.beam_depth - 1]])

        best_node = beam[self.beam_depth - 1][idx_best_node]
        while best_node.parent != beam[0][0]:
            best_node = best_node.parent

        return best_node.move

    def evaluate_board(self, board: chess.Board) -> float:
        return eval_board(self.model, board)

    def __str__(self) -> str:
        return "DLAgent"
