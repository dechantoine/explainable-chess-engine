import unittest

import chess.pgn
import torch

from src.engine.agents.dl_agent import DLAgent
from src.engine.agents.stockfish_agent import StockfishAgent
from src.engine.games import Game


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=12 * 8 * 8, out_features=1)

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        return self.linear(x)

class GameTestCase(unittest.TestCase):
    def setUp(self):
        self.fen = "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"
        self.board = chess.Board(fen=self.fen)

        self.model = MockModel()

    def test_init(self):
        stockfish_agent = StockfishAgent(is_white=True)
        dl_agent = DLAgent(model=self.model, is_white=True)

        with self.assertRaises(ValueError):
            Game(player_1=stockfish_agent, player_2=dl_agent, board=None)

        dl_agent = DLAgent(model=self.model, is_white=False)
        game = Game(player_1=stockfish_agent, player_2=dl_agent, board=self.board)

        self.assertIsInstance(game.board, chess.Board)
        self.assertIsInstance(game.n_moves, int)
        self.assertGreater(game.n_moves, 0)
        assert game.current_player == dl_agent

    def test_forward_one_half_move(self):
        stockfish_agent = StockfishAgent(is_white=True)
        dl_agent = DLAgent(model=self.model, is_white=False)
        game = Game(player_1=stockfish_agent, player_2=dl_agent, board=self.board)

        n_moves = game.n_moves
        game.forward_one_half_move()

        self.assertEqual(game.n_moves, n_moves + 1)
        self.assertEqual(game.current_player, stockfish_agent)

    def test_backward_one_half_move(self):
        stockfish_agent = StockfishAgent(is_white=True)
        dl_agent = DLAgent(model=self.model, is_white=False)
        game = Game(player_1=stockfish_agent, player_2=dl_agent, board=self.board)

        with self.assertRaises(IndexError):
            game.backward_one_half_move()

        game.forward_one_half_move()
        n_moves = game.n_moves
        game.backward_one_half_move()
        self.assertEqual(game.n_moves, n_moves - 1)
        self.assertEqual(game.current_player, dl_agent)

    def test_play(self):
        stockfish_agent = StockfishAgent(is_white=True)
        dl_agent = DLAgent(model=self.model, is_white=False)
        game = Game(player_1=stockfish_agent, player_2=dl_agent, board=None)

        winner, termination, n_moves = game.play()

        self.assertIsInstance(winner, bool)
        self.assertIsInstance(termination, chess.Termination)
        self.assertIsInstance(n_moves, int)
