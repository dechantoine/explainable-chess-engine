import datetime

import chess.pgn
import numpy as np
from loguru import logger
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.engine.agents.base_agent import BaseAgent
from src.engine.agents.stockfish_agent import StockfishAgent


class Game:
    def __init__(self, player_1: BaseAgent, player_2: BaseAgent, board: chess.Board = None) -> None:
        """Initialize the game with two players and a board.

        Args:
            player_1 (BaseAgent): the first player
            player_2 (BaseAgent): the second player
            board (chess.Board): the board of the game

        """
        if player_1.is_white and not player_2.is_white:
            self.whites = player_1
            self.blacks = player_2
        elif player_2.is_white and not player_1.is_white:
            self.whites = player_2
            self.blacks = player_1
        else:
            raise ValueError("One player must be white and the other black.")

        self.current_player = self.whites
        self.board = board

        if board is None:
            self.board = chess.Board()
        else:
            if not self.board.turn:
                self.current_player = self.blacks

        self.n_moves = self.board.ply()

        self.result = {}

    def forward_one_half_move(self) -> None:
        """Play one half move forward."""
        move = self.current_player.next_move(self.board)
        self.n_moves += 1
        self.board.push(move)
        self.current_player = (
            self.whites if self.current_player == self.blacks else self.blacks
        )

    def backward_one_half_move(self) -> None:
        """Delete latest half move."""
        try:
            self.board.pop()
        except IndexError:
            raise IndexError("Moves stack is empty because board has been reset or initialized from FEN")
        else:
            self.n_moves -= 1
            self.current_player = (
                self.whites if self.current_player == self.blacks else self.blacks
            )

    def play(self) -> (str, BaseAgent, chess.Termination, int, list[chess.Move]):
        """Play the game with current player until termination.

        Returns:
            result (dict): dict containing all game information

        """
        outcome = self.board.outcome()
        while not outcome:
            self.forward_one_half_move()
            outcome = self.board.outcome()

        self.result["result"] = outcome.result()

        if outcome.winner:
            self.result["winner"] = str(self.whites)
        elif outcome.winner is None:
            self.result["winner"] = "tie"
        else:
            self.result["winner"] = str(self.blacks)

        self.result["termination"] = outcome.termination
        self.result["move_stack"] = self.board.move_stack
        self.result["n_moves"] = self.n_moves

        return self.result


class Match:
    def __init__(self, player_1: BaseAgent, player_2: BaseAgent, n_games: int = 50) -> None:
        """Initialize a match of n games between 2 players.

        Args:
            player_1 (BaseAgent): the first player
            player_2 (BaseAgent): the second player
            n_games (int): number of games to play

        """
        self.player_1 = player_1
        self.player_2 = player_2

        if n_games == 0:
            raise ValueError("n_games have to be non-zero")
        self.n_games = n_games

        rng = np.random.default_rng()
        self.whites = np.array([player_1, player_2] * (n_games // 2))
        rng.shuffle(self.whites)
        if n_games % 2 == 1:
            self.whites = np.append(self.whites, np.random.choice(a=[player_1, player_2], size=1))
        self.whites = self.whites.tolist()

        self.results = []

    def play(self) -> None:
        """Play the match."""
        for white in tqdm(iterable=self.whites,
                          desc=f"Playing {self.n_games} games between {str(self.player_1)} and {str(self.player_2)}..."):
            white.set_color(is_white=True)
            if white == self.player_1:
                self.player_2.set_color(is_white=False)
            else:
                self.player_1.set_color(is_white=False)

            game = Game(player_1=self.player_1, player_2=self.player_2)

            result = game.play()
            self.results.append(result)

    def parallel_play(self, max_workers: int = 8) -> None:
        """Play the match in parallel.

        Args:
            max_workers (int): number of workers to use

        """
        outcomes = process_map(
            star_play,
            [(white, self.player_1, self.player_2) for white in self.whites],
            max_workers=max_workers,
            chunksize=max(1, len(self.whites) // 100),
            desc=f"Playing {self.n_games} games between {str(self.player_1)} and {str(self.player_2)} in parallel...",
        )

        self.results = outcomes

    def save(self, filename: str) -> None:
        """Save the results of the match to a PGN file.

        Args:
            filename (str): the name of the file

        """
        # ensure that the file is empty
        open(filename, "w").close()

        for i, white in enumerate(self.whites):
            black = self.player_1 if white == self.player_2 else self.player_2

            board = chess.Board()
            for move in self.results[i]["move_stack"]:
                board.push(move)

            pgn = chess.pgn.Game.from_board(board)
            pgn.headers["Event"] = f"Match between {str(self.player_1)} and {str(self.player_2)}"
            pgn.headers["Site"] = ""
            pgn.headers["Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pgn.headers["Round"] = str(i + 1)
            pgn.headers["White"] = str(white)
            pgn.headers["Black"] = str(black)
            pgn.headers["Result"] = self.results[i]["result"]
            pgn.headers["WhiteElo"] = str(white.elo) if hasattr(white, "elo") else "?"
            pgn.headers["BlackElo"] = str(black.elo) if hasattr(black, "elo") else "?"

            with open(filename, "a") as f:
                f.write(str(pgn) + "\n\n")


def star_play(args: tuple) -> tuple:
    """Prepare the arguments for the play method."""
    white = args[0]
    white.set_color(is_white=True)
    if args[0] == args[1]:
        black = args[2]
    else:
        black = args[1]
    black.set_color(is_white=False)

    game = Game(player_1=white, player_2=black)
    return game.play()


class EloEvaluator:
    def __init__(self, player: BaseAgent, resolution: int = 50, stockfish_min_elo: int = 1300,
                 parallel: bool = True) -> None:
        """Initialize the EloEvaluator with a player to evaluate.

        Args:
            player(BaseAgent): the player to evaluate
            resolution (int): the resolution of the Elo rating
            stockfish_min_elo (int): the minimum Elo rating of Stockfish
            parallel (bool): whether to play matches in parallel

        """
        self.player_1 = player
        self.resolution = resolution
        self.parallel = parallel
        self.elo = None
        self.current_stockfish = stockfish_min_elo
        self.matches = {}

    def _play_match(self, n_games: int = 5) -> float:
        """Play a match between the player and Stockfish.

        Args:
            n_games (int): the number of games to play

        Returns:
            win_rate (float): the win rate of the player

        """
        stockfish = StockfishAgent(is_white=False, elo=self.current_stockfish)
        match = Match(player_1=self.player_1, player_2=stockfish, n_games=n_games)
        match.parallel_play() if self.parallel else match.play()

        results = [r["winner"] for r in match.results]

        win_score = []
        for r in results:
            if r == str(self.player_1):
                win_score.append(1)
            elif r == str(stockfish):
                win_score.append(0)
            else:
                win_score.append(0.5)

        win_rate = np.mean(win_score)

        self.matches[self.current_stockfish] = {"match": match,
                                                "win_rate": win_rate}

        return win_rate

    def _compute_elo(self) -> float:
        """Compute the Elo rating of the player.

        Returns:
            elo (float): the Elo rating

        """
        last_win_rate = self.matches[self.current_stockfish]["win_rate"]
        self.elo = self.current_stockfish - self.resolution * (1 - last_win_rate)
        return self.elo

    def evaluate(self) -> float:
        """Evaluate the player's Elo rating.

        Returns:
            elo (float): the Elo rating

        """
        logger.info(f"Evaluating {str(self.player_1)}'s Elo rating, starting with Stockfish {self.current_stockfish}...")
        current_win_rate = 1
        while current_win_rate > 0.5:
            current_win_rate = self._play_match()
            logger.info(f"Win rate against Stockfish {self.current_stockfish} : {current_win_rate}")
            self.current_stockfish += self.resolution
            logger.info(f"Updating Stockfish Elo to {self.current_stockfish}")
        self.current_stockfish -= self.resolution

        self.elo = self._compute_elo()
        logger.info(f"{str(self.player_1)}'s Elo rating: {self.elo}")
        return self.elo
