import os
import sys
from typing import Dict

import numpy as np
import torch
from anytree.exporter import DictExporter
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..src.data.data_utils import clean_board
from ..src.engine.agents.dl_agent import choose_move_from_beam
from ..src.engine.agents.policies import Strategy, beam_search, eval_board, eval_boards, one_depth_eval
from ..src.models.multi_input_conv import MultiInputConv

CHKPT = "checkpoint.pt"


class Board(BaseModel):
    board: str = Field(description='A chess board represented either in FEN or PGN',
                       examples=['rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2',
                                 '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O'])


class BoardEval(BaseModel):
    board: str = Field(description='A chess board represented either in FEN or PGN',
                       examples=['rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2',
                                 '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O'])
    evaluation: float = Field(description='Evaluation of the board',
                              examples=[0.5, -0.3])


class OneMoveEval(BaseModel):
    move: str = Field(description='A chess move in UCI format',
                      examples=['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6', 'e1g1', 'e8e7'])
    board: str = Field(description='A chess board represented either in FEN or PGN',
                       examples=['rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2',
                                 '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O'])
    evaluation: float = Field(description='Evaluation of the board after the move',
                              examples=[0.5, -0.3])


class BeamSearch(BaseModel):
    board: str = Field(description='A chess board represented either in FEN or PGN',
                       examples=['rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2',
                                 '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O'])
    depth: int = Field(description='Depth of the search',
                       examples=[3, 5])
    width: int = Field(description='Width of the search',
                       examples=[2, 3])
    player_strategy: Strategy = Field(description='Strategy to use for selecting moves of the current player',
                                      examples=[Strategy.GREEDY, Strategy.TOP_K])
    opponent_strategy: Strategy = Field(description='Strategy to use for selecting moves of the opponent',
                                        examples=[Strategy.GREEDY, Strategy.TOP_K])
    player_top_k: int = Field(description='Number of top moves to consider for the current player',
                              examples=[1, 2])
    opponent_top_k: int = Field(description='Number of top moves to consider for the opponent',
                                examples=[1, 2])


file = sys.argv[0]
DIR_PATH = 'api'

chkpt = torch.load(os.path.join(DIR_PATH, f"checkpoints/{CHKPT}"))
model = MultiInputConv()
model.load_state_dict(state_dict=chkpt["model_state_dict"])
model.eval()

app = FastAPI()


@app.get("/")
async def root():
    return str(model)


@app.post('/evaluate')
async def get_evaluate_board(input: Board) -> BoardEval:
    """Evaluate the board.

    Args:
        input (BoardEval): board to evaluate

    Returns:
        BoardEval: board evaluation

    """
    board = clean_board(board=input.board)
    return BoardEval(board=board.fen(), evaluation=eval_board(model=model, board=board))


@app.post('/one_depth_eval')
async def get_one_move_eval(input: Board) -> list[OneMoveEval]:
    """Get all possible boards in one move with their evaluation, sorted by best move for current player.

    Args:
        input (BoardEval): board to consider

    Returns:
        list[OneMoveEval]: list of possible boards with their evaluation.

    """
    board = clean_board(board=input.board)

    legal_boards, legal_moves, scores = one_depth_eval(
        model=model, boards=[board], min_score=-100, max_score=100
    )

    # get scores argsort
    argsort = np.argsort(scores[0])
    if board.turn:
        argsort = argsort[::-1]

    scores = np.array(scores[0])[argsort]
    legal_boards = np.array(legal_boards[0])[argsort]
    legal_moves = np.array(legal_moves[0])[argsort]

    output = [OneMoveEval(
        move=str(legal_moves[i]),
        board=legal_boards[i].fen(),
        evaluation=float(scores[i])
    ) for i in range(len(scores))]

    return output


@app.post('/beam_search')
async def get_beam_search(input: BeamSearch) -> Dict:
    """Get best moves using beam search.

    Args:
        input (BeamSearch): input parameters for beam search

    Returns:
        BeamTree: tree of boards and moves and evaluations

    """
    board = clean_board(board=input.board)
    beam = beam_search(
        model=model,
        board=board,
        depth=input.depth,
        beam_width=input.width,
        player_strategy=input.player_strategy,
        opponent_strategy=input.opponent_strategy,
        player_top_k=input.player_top_k,
        opponent_top_k=input.opponent_top_k
    )

    beam.board = beam.board.fen()
    beam.move = str(beam.move)

    for node in beam.descendants:
        node.board = node.board.fen()
        node.move = str(node.move)

    exporter = DictExporter()
    return exporter.export(beam)


@app.post('/best_move_from_beam')
async def get_best_move_from_beam(input: BeamSearch) -> str:
    """Get best move from beam search.

    Args:
        input (BeamSearch): input parameters for beam search

    Returns:
        str: best move

    """
    board = clean_board(board=input.board)
    beam = beam_search(
        model=model,
        board=board,
        depth=input.depth,
        beam_width=input.width,
        player_strategy=input.player_strategy,
        opponent_strategy=input.opponent_strategy,
        player_top_k=input.player_top_k,
        opponent_top_k=input.opponent_top_k
    )

    return str(choose_move_from_beam(beam=beam, is_white=board.turn, gamma=0.9, max_score=100, min_score=-100))


@app.post('/batch/evaluate')
async def batch_evaluate_board(input: list[Board]) -> list[BoardEval]:
    """Evaluate the boards.

    Args:
        input (list[BoardEval]): boards to evaluate

    Returns:
        list[BoardEval]: board evaluations

    """
    boards = [clean_board(board=inp.board) for inp in input]
    evaluations = eval_boards(model=model, boards=boards)
    return [BoardEval(board=board.fen(), evaluation=evaluation) for board, evaluation in zip(boards, evaluations)]


@app.post('/batch/one_depth_eval')
async def batch_one_move_eval(input: list[Board]) -> list[list[OneMoveEval]]:
    """Get all possible boards in one move with their evaluation, sorted by best move for current player.

    Args:
        input (list[BoardEval]): boards to consider

    Returns:
        list[list[OneMoveEval]]: list of possible boards with their evaluation.

    """
    boards = [clean_board(board=inp.board) for inp in input]

    legal_boards, legal_moves, scores = one_depth_eval(
        model=model, boards=boards, min_score=-100, max_score=100
    )

    # get scores argsort
    argsorts = [np.argsort(scores[i]) for i in range(len(scores))]
    for i, board in enumerate(boards):
        if board.turn:
            argsorts[i] = argsorts[i][::-1]

    output = []
    for i in range(len(scores)):
        scores_i = np.array(scores[i])[argsorts[i]]
        legal_boards_i = np.array(legal_boards[i])[argsorts[i]]
        legal_moves_i = np.array(legal_moves[i])[argsorts[i]]

        output.append([OneMoveEval(
            move=str(legal_moves_i[j]),
            board=legal_boards_i[j].fen(),
            evaluation=float(scores_i[j])
        ) for j in range(len(scores_i))])

    return output
