import numpy as np

class Agent():
    def __init__(self, board, policy):
        self.board = board
        self.policy = policy
        self.is_white = None
        
    def set_color(self, is_white):
        self.is_white = is_white
        
    def next_move(self):
        return self.policy(self.board)