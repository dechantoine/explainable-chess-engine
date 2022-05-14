import chess
import numpy as np

class Game():
    def __init__(self, board, player_1, player_2, enforced_colors=True):
        self.board = board
        
        if enforced_colors:
            self.whites = player_1
            self.blacks = player_2
        else:
            self.whites = np.random.choice([player_1, player_2])
            self.blacks = (player_1 if self.whites==player_2 else player_2)
            
        self.whites.set_color(True)
        self.blacks.set_color(False)
        
        self.current_player = self.whites
            
            
    def forward_one_half_move(self):
        move = self.current_player.next_move()
        self.board.push(move)
        self.current_player = (self.whites if self.current_player==self.blacks else self.blacks)
        
    def backward_one_half_move(self):
        self.board.pop()
        self.current_player = (self.whites if self.current_player==self.blacks else self.blacks)