import unittest
from test.mock_torch_model import MockModel

import chess.pgn
import numpy as np
from anytree import AnyNode, LevelOrderGroupIter

from src.engine.agents.policies import (
    beam_sampling,
    beam_search,
    eval_board,
    get_legal_moves,
    one_depth_eval,
    push_legal_moves,
)


class PoliciesTestCase(unittest.TestCase):
    def setUp(self):
        self.fen = "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"
        self.checkmate_fen = " 1r3rk1/p3bppp/1pb5/2p1p1B1/3P2n1/2P3P1/PP2PPqP/R2QR1K1 w - - 0 16"
        self.model = MockModel()
        self.board = chess.Board(fen=self.fen)
        self.checkmate_board = chess.Board(fen=self.checkmate_fen)

        self.beam_depth = 8
        self.beam_width = 4

    def test_eval_board(self):
        score = eval_board(model=self.model, board=self.board)

        self.assertIsInstance(score, float)

    def test_get_legal_moves(self):
        legal_moves = get_legal_moves(boards=[self.board, self.board.copy()])

        self.assertIsInstance(legal_moves, list)
        self.assertIsInstance(legal_moves[0], list)
        self.assertIsInstance(legal_moves[0][0], chess.Move)

        self.assertEqual(len(legal_moves), 2)
        self.assertEqual(len(legal_moves[0]), 28)

    def test_push_legal_moves(self):
        legal_moves = get_legal_moves(boards=[self.board, self.board.copy()])
        pushed_boards = push_legal_moves(
            boards=[self.board, self.board.copy()], legal_moves=legal_moves
        )

        self.assertIsInstance(pushed_boards, list)
        self.assertIsInstance(pushed_boards[0], list)
        self.assertIsInstance(pushed_boards[0][0], chess.Board)

        self.assertEqual(len(pushed_boards), 2)
        self.assertEqual(len(pushed_boards[0]), 28)

    def test_one_depth_eval(self):
        boards = [self.board, self.board.copy()]

        legal_boards, legal_moves, scores = one_depth_eval(
            model=self.model, boards=boards
        )

        self.assertIsInstance(legal_boards, list)
        self.assertIsInstance(legal_boards[0], list)
        self.assertIsInstance(legal_boards[0][0], chess.Board)

        self.assertIsInstance(legal_moves, list)
        self.assertIsInstance(legal_moves[0], list)
        self.assertIsInstance(legal_moves[0][0], chess.Move)

        self.assertIsInstance(scores, list)
        self.assertIsInstance(scores[0], list)
        self.assertIsInstance(scores[0][0], np.float32)

        legal_boards, legal_moves, scores = one_depth_eval(
            model=self.model, boards=[self.checkmate_board, self.board]
        )

        self.assertEqual(len(legal_boards[0]), 1)
        self.assertEqual(len(legal_moves[0]), 1)
        self.assertEqual(len(scores[0]), 1)

        self.assertEqual(legal_boards[0][0], self.checkmate_board)
        self.assertIsNone(legal_moves[0][0])
        self.assertEqual(scores[0][0], -1)

    def test_beam_sampling(self):
        beam = beam_sampling(
            boards=[[self.board.copy() for _ in range(20)]],
            scores=[np.linspace(-1, 1, 20).tolist()],
            moves=[[next(self.board.generate_legal_moves()) for _ in range(20)]],
            beam_width=self.beam_width,
            is_white=True,
            is_opponent=False,
        )

        self.assertIsInstance(beam, list)
        self.assertIsInstance(beam[0], dict)

        self.assertIsInstance(beam[0]["board"], chess.Board)
        self.assertIsInstance(beam[0]["move"], chess.Move)
        self.assertIsInstance(beam[0]["score"], float)

        self.assertEqual(len(beam), self.beam_width)
        self.assertEqual(beam[0]["score"], 1)

        beam = beam_sampling(
            boards=[[self.board.copy() for _ in range(20)]],
            scores=[np.linspace(-1, 1, 20).tolist()],
            moves=[[next(self.board.generate_legal_moves()) for _ in range(20)]],
            beam_width=self.beam_width,
            is_white=True,
            is_opponent=True,
        )

        self.assertEqual(len(beam), 1)
        self.assertEqual(beam[0]["score"], 1)

        beam = beam_sampling(
            boards=[[self.board.copy() for _ in range(20)]],
            scores=[np.linspace(-1, 1, 20).tolist()],
            moves=[[next(self.board.generate_legal_moves()) for _ in range(20)]],
            beam_width=self.beam_width,
            is_white=False,
            is_opponent=True,
        )

        self.assertEqual(beam[0]["score"], -1)

    def test_beam_search(self):
        beam = beam_search(
            model=self.model,
            board=self.board,
            depth=self.beam_depth,
            beam_width=self.beam_width,
        )

        self.assertIsInstance(beam, AnyNode)
        self.assertEqual(beam.height, self.beam_depth)

        for depth in list(LevelOrderGroupIter(beam))[1:]:
            self.assertEqual(len(depth), self.beam_width)

        self.assertIsInstance(beam.children[0].score, np.float32)
        self.assertIsInstance(beam.children[0].board, chess.Board)
        self.assertIsInstance(beam.children[0].move, chess.Move)
