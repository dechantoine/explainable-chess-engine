import chess.pgn

from src.engine.agents.base_agent import BaseAgent


class Game:
    def __init__(self, player_1: BaseAgent, player_2: BaseAgent, board: chess.Board = None) -> None:
        """Initialize the game with two players and a board."""
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

    def play(self) -> (bool, chess.Termination, int):
        """Play the game with current player until termination.

        Returns:
            winner (bool): the winner of the game or None if drawn
            termination (str): the termination reason
            n_moves (int): the number of half moves of the game

        """
        outcome = self.board.outcome()
        while not outcome:
            self.forward_one_half_move()
            outcome = self.board.outcome()

        self.winner = outcome.winner
        self.termination = outcome.termination

        return self.winner, self.termination, self.n_moves
