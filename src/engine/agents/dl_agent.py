import hashlib

import chess.pgn
import numpy as np
import torch
from anytree import LevelOrderGroupIter, LevelOrderIter, findall_by_attr
from loguru import logger

from src.engine.agents.base_agent import BaseAgent
from src.engine.agents.policies import beam_search, eval_board


class DLAgent(BaseAgent):
    def __init__(self,
                 model: torch.nn.Module,
                 is_white: bool,
                 beam_width: int = 5,
                 beam_depth: int = 11,
                 beam_player_strategy: str = "greedy",
                 beam_opponent_strategy: str = "greedy",
                 beam_player_top_k: int = 1,
                 beam_opponent_top_k: int = 1,
                 name: str = None
                 ) -> None:

        super().__init__(is_white)
        self.model = model
        self.model.eval()
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.beam_player_strategy = beam_player_strategy
        self.beam_opponent_strategy = beam_opponent_strategy
        self.beam_player_top_k = beam_player_top_k
        self.beam_opponent_top_k = beam_opponent_top_k

        self.name = name

        self.set_min_max_score()

    def set_min_max_score(self):
        self.max_score = 100
        self.min_score = -100

    @logger.catch(level="DEBUG")
    def next_move(self, board: chess.Board) -> chess.Move:
        # beam = list(LevelOrderGroupIter(beam_search(self.model, board, self.beam_depth, self.beam_width, self.min_score, self.max_score)))
        beam = beam_search(model=self.model,
                           board=board,
                           depth=self.beam_depth,
                           beam_width=self.beam_width,
                           player_strategy=self.beam_player_strategy,
                           opponent_strategy=self.beam_opponent_strategy,
                           opponent_top_k=self.beam_opponent_top_k,
                           min_score=self.min_score,
                           max_score=self.max_score)

        beam_mean_score = np.mean([node.score for node in list(LevelOrderIter(beam))][1:])

        # first check if an immediate move leads to checkmate or tie
        for node in list(LevelOrderGroupIter(beam))[1]:
            if node.board.outcome():
                if self.is_white:
                    if node.board.outcome().result() == "1-0":
                        logger.info(f"Immediate checkmate for {self}")
                        return node.move
                    if node.board.outcome().result() == "1/2-1/2" and beam_mean_score < 0:
                        logger.info(f"Immediate tie for {self}")
                        return node.move
                else:
                    if node.board.outcome().result() == "0-1":
                        logger.info(f"Immediate checkmate for {self}")
                        return node.move
                    if node.board.outcome().result() == "1/2-1/2" and beam_mean_score > 0:
                        logger.info(f"Immediate tie for {self}")
                        return node.move


        # tag each leaf node with "finished" attribute if it is at max depth or checkmate for the player
        max_depth = max([node.depth for node in list(LevelOrderIter(beam))])
        for node in list(LevelOrderIter(beam)):
            if node.depth == max_depth:
                node.finished = True
            elif node.board.outcome():
                if node.board.outcome().result() == "1-0" and self.is_white:
                    node.finished = True
                elif node.board.outcome().result() == "0-1" and not self.is_white:
                    node.finished = True

        # identify moves with most finishing nodes
        nb_finished_nodes = [len(findall_by_attr(name="finished", value=True, node=node))
                             for node in list(LevelOrderGroupIter(beam))[1]]

        # TODO: implement a tie-breaker
        best_node = list(LevelOrderGroupIter(beam))[1][np.argmax(nb_finished_nodes)]

        return best_node.move

    def evaluate_board(self, board: chess.Board) -> float:
        return eval_board(self.model, board)

    def __str__(self) -> str:
        if self.name:
            return self.name
        # get list of self parameters
        hash = hashlib.md5(
            (str(self.model.model_hash()) +
             str(self.beam_width) +
             str(self.beam_depth) +
             str(self.beam_player_strategy) +
             str(self.beam_player_top_k) +
             str(self.beam_opponent_top_k)
             ).encode()
        ).hexdigest()
        return f"DLAgent-{hash}"
