import os
import unittest
from test.mock_torch_model import MockModel

import chess.pgn

from src.engine.agents.base_agent import BaseAgent
from src.engine.agents.dl_agent import DLAgent
from src.engine.agents.stockfish_agent import StockfishAgent
from src.engine.games import EloEvaluator, Game, Match


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
        self.assertIsInstance(game.result, dict)
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

        result = game.play()

        self.assertIsInstance(result["result"], str)
        self.assertIsInstance(result["winner"], str)
        self.assertIsInstance(result["termination"], chess.Termination)
        self.assertIsInstance(result["n_moves"], int)
        self.assertIsInstance(result["move_stack"], list)


class MatchTestCase(unittest.TestCase):
    def setUp(self):
        model = MockModel()
        self.stockfish_agent = StockfishAgent(is_white=True)
        self.dl_agent = DLAgent(model=model, is_white=True)

    def tearDown(self):
        if os.path.exists("test/test.pgn"):
            os.remove("test/test.pgn")

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
        assert all(isinstance(match.results[i]["result"], str) for i in range(5))
        assert all(isinstance(match.results[i]["winner"], str) for i in range(5))
        assert all(isinstance(match.results[i]["termination"], chess.Termination) for i in range(5))
        assert all(isinstance(match.results[i]["n_moves"], int) for i in range(5))
        assert all(isinstance(match.results[i]["move_stack"], list) for i in range(5))

    def test_parallel_play(self):
        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=5)
        match.parallel_play()

        self.assertEqual(len(match.results), 5)
        assert all(isinstance(match.results[i]["result"], str) for i in range(5))
        assert all(isinstance(match.results[i]["winner"], str) for i in range(5))
        assert all(isinstance(match.results[i]["termination"], chess.Termination) for i in range(5))
        assert all(isinstance(match.results[i]["n_moves"], int) for i in range(5))
        assert all(isinstance(match.results[i]["move_stack"], list) for i in range(5))


    def test_save_pgn(self):
        match = Match(player_1=self.stockfish_agent, player_2=self.dl_agent, n_games=2)
        match.play()
        match.save("test/test.pgn")

        pgn_match = open("test/test.pgn")
        first_game = chess.pgn.read_game(pgn_match)
        second_game = chess.pgn.read_game(pgn_match)
        self.assertIsInstance(first_game, chess.pgn.Game)
        self.assertIsInstance(second_game, chess.pgn.Game)

        third_game = chess.pgn.read_game(pgn_match)
        self.assertIsNone(third_game)


class EloEvaluatorTestCase(unittest.TestCase):
    def setUp(self):
        model = MockModel()
        self.dl_agent = DLAgent(model=model, is_white=True)
        self.win_rates = {1300: {"match": Match(player_1=self.dl_agent, player_2=self.dl_agent, n_games=5),
                                 "win_rate": 0.9},
                          1400: {"match": Match(player_1=self.dl_agent, player_2=self.dl_agent, n_games=5),
                                 "win_rate": 0.7},
                          1500: {"match": Match(player_1=self.dl_agent, player_2=self.dl_agent, n_games=5),
                                 "win_rate": 0.4},
                          }

    def test_init(self):
        elo_evaluator = EloEvaluator(player=self.dl_agent)
        self.assertIsInstance(elo_evaluator.player_1, BaseAgent)
        self.assertIsInstance(elo_evaluator.matches, dict)

    def test_play_match(self):
        elo_evaluator = EloEvaluator(player=self.dl_agent)
        win_rate = elo_evaluator._play_match(n_games=5)

        self.assertEqual(len(elo_evaluator.matches), 1)
        assert (isinstance(elo_evaluator.matches[1300]["match"], Match))
        self.assertIsInstance(elo_evaluator.matches[1300]["win_rate"], float)
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)

    def test_compute_elo(self):
        elo_evaluator = EloEvaluator(player=self.dl_agent, resolution=100)
        elo_evaluator.matches = self.win_rates
        elo_evaluator.current_stockfish = 1500
        elo_evaluator._compute_elo()

        self.assertIsInstance(elo_evaluator.elo, float)
        self.assertEqual(elo_evaluator.elo, 1440)
