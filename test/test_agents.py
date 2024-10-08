import unittest
from test.mock_torch_model import MockModel

import chess.pgn

from src.engine.agents.dl_agent import DLAgent
from src.engine.agents.stockfish_agent import StockfishAgent


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.fen = "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"
        self.board = chess.Board(fen=self.fen)

        model = MockModel()

        self.stockfish_agent = StockfishAgent(is_white=True)
        self.dl_agent = DLAgent(model=model, is_white=True)


    def test_evaluate_board(self):
        stockfish_score = self.stockfish_agent.evaluate_board(self.board)
        dl_score = self.dl_agent.evaluate_board(self.board)

        self.assertIsInstance(stockfish_score, float)
        self.assertIsInstance(dl_score, float)


    def test_next_move(self):
        stockfish_move = self.stockfish_agent.next_move(self.board)
        dl_move = self.dl_agent.next_move(self.board)

        self.assertIsInstance(stockfish_move, chess.Move)
        self.assertIsInstance(dl_move, chess.Move)
