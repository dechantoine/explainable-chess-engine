import datetime

import chess.pgn
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.engine.agents.base_agent import BaseAgent


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
            result (str): the result in pgn format
            winner (BaseAgent): the winning Agent of the game or None if drawn
            termination (str): the termination reason
            n_moves (int): the number of half moves of the game
            move_stack (list[chess.move]): moves played

        """
        outcome = self.board.outcome()
        while not outcome:
            self.forward_one_half_move()
            outcome = self.board.outcome()

        if outcome.winner:
            self.winner = self.whites
        elif outcome.winner is None:
            self.winner = None
        else:
            self.winner = self.blacks

        self.termination = outcome.termination
        self.move_stack = self.board.move_stack

        return outcome.result(), self.winner, self.termination, self.n_moves, self.move_stack


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

            result, winner, termination, n_moves, move_stack = game.play()
            self.results.append((result, winner, termination, n_moves, move_stack))

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
            for move in self.results[i][4]:
                board.push(move)

            pgn = chess.pgn.Game.from_board(board)
            pgn.headers["Event"] = f"Match between {str(self.player_1)} and {str(self.player_2)}"
            pgn.headers["Site"] = ""
            pgn.headers["Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pgn.headers["Round"] = str(i + 1)
            pgn.headers["White"] = str(white)
            pgn.headers["Black"] = str(black)
            pgn.headers["Result"] = self.results[i][0]
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
