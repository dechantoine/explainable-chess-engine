import unittest

import chess.pgn
import torch

from src.engine.agents.base_agent import BaseAgent
from src.engine.agents.dl_agent import DLAgent
from src.engine.agents.stockfish_agent import StockfishAgent
from src.engine.games import Game, Match


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

        winner, termination, n_moves, move_stack = game.play()

        self.assertIsInstance(winner, BaseAgent)
        self.assertIsInstance(termination, chess.Termination)
        self.assertIsInstance(n_moves, int)
        self.assertIsInstance(move_stack, list)


class MatchTestCase(unittest.TestCase):
    def setUp(self):
        model = MockModel()
        self.stockfish_agent = StockfishAgent(is_white=True)
        self.dl_agent = DLAgent(model=model, is_white=True)

    def test_init(self):
        with self.assertRaises(ValueError):
            Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=0)

        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=10)
        self.assertIsInstance(match.whites, list)
        self.assertEqual(len(match.whites), 10)
        self.assertCountEqual(match.whites, [self.stockfish_agent, self.dl_agent] * 5)
        self.assertEqual(len(match.results), 0)

        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=11)
        self.assertCountEqual(match.whites[:-1], [self.stockfish_agent, self.dl_agent] * 5)

    def test_play(self):
        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=5)
        match.play()

        self.assertEqual(len(match.results), 5)
        assert all(isinstance(match.results[i][0], BaseAgent) for i in range(5))
        assert all(isinstance(match.results[i][1], chess.Termination) for i in range(5))
        assert all(isinstance(match.results[i][2], int) for i in range(5))
        assert all(isinstance(match.results[i][3], list) for i in range(5))

    def test_parallel_play(self):
        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=5)
        match.parallel_play()

        self.assertEqual(len(match.results), 5)
        assert all(isinstance(match.results[i][0], BaseAgent) for i in range(5))
        assert all(isinstance(match.results[i][1], chess.Termination) for i in range(5))
        assert all(isinstance(match.results[i][2], int) for i in range(5))
