import os

import chess.pgn
import numpy as np
import pandas as pd
import torch
from anytree import LevelOrderGroupIter
from loguru import logger
from tqdm.contrib.concurrent import process_map

from src.data.data_utils import read_boards_from_pgn
from src.engine.agents.policies import beam_search
from src.models.simple_feed_forward import SimpleFF

end_opening = 10
end_midgame = 10
max_depth = 10
max_width = 10
MODELS_DIR = "models_checkpoint/simple_ff_0"
CHKPT = "checkpoint_36700.pt"
GAMES_DIR = "test/test_data"
RESULTS_DIR = "studies/results"


@logger.catch
def beam_benchmark_board(board: chess.Board, model: torch.nn.Module, max_depth=10, max_width=10,
                         return_absolute: bool = True) -> tuple:
    """Benchmark the beam search algorithm on a board.

    Args:
        board (chess.Board): board to evaluate
        model (torch.nn.Module): model to evaluate the board
        max_depth (int): maximum depth of the tree
        max_width (int): maximum width of the tree
        return_absolute (bool): return the absolute score or relative to the best score

    Returns:
        tuple: greedy_scores, mean_scores

    """
    greedy_scores = np.zeros(shape=(max_depth, max_width))
    mean_scores = np.zeros(shape=(max_depth, max_width))

    for width in range(1, max_width + 1):
        beam = list(LevelOrderGroupIter(beam_search(model=model, board=board, depth=max_depth * 2, beam_width=width)))

        for depth in range(1, max_depth * 2, 2):
            greedy_scores[depth // 2, width - 1] = max([node.score for node in beam[depth]])
            mean_scores[depth // 2, width - 1] = np.mean([node.score for node in beam[depth]])


    if not return_absolute:
        if board.turn:
            greedy_scores = np.abs(greedy_scores - greedy_scores.max())
            mean_scores = np.abs(mean_scores - mean_scores.max())
        else:
            greedy_scores -= greedy_scores.min()
            mean_scores -= mean_scores.min()

    return greedy_scores, mean_scores


@logger.catch
def star_beam_benchmark_board(args: tuple) -> tuple:
    """Benchmark the beam search algorithm on a board.

    Args:
        args (tuple): arguments for the function

    Returns:
        tuple: greedy_scores, mean_scores

    """
    return beam_benchmark_board(*args)


@logger.catch
def beam_benchmark_boards(boards: list[chess.Board], model: torch.nn.Module, max_depth: int = 10, max_width: int = 10,
                          max_workers: int = 1) -> tuple:
    """Benchmark the beam search algorithm on a list of boards.

    Args:
        boards (list[chess.Board]): boards to evaluate
        model (torch.nn.Module): model to evaluate the boards
        max_depth (int): maximum depth of the tree
        max_width (int): maximum width of the tree
        max_workers (int): number of workers to use

    Returns:
        tuple: greedy_scores, mean_scores

    """
    benchs = process_map(
        star_beam_benchmark_board,
        [(board, model, max_depth, max_width, True) for board in boards],
        max_workers=max_workers,
        chunksize=max(1, len(boards) // (100 * max_workers)),
        desc=f"Benchmarking beam search for max depth = {max_depth} and max width = {max_width} with {max_workers} workers...",
    )

    greedy_scores, mean_scores = zip(*benchs)

    greedy_scores = np.array(greedy_scores)
    mean_scores = np.array(mean_scores)

    greedy_scores = greedy_scores.mean(axis=0)
    mean_scores = mean_scores.mean(axis=0)

    return greedy_scores, mean_scores


if __name__ == "__main__":
    # Load the model
    chkpt = torch.load(os.path.join(MODELS_DIR, CHKPT))
    model = SimpleFF()
    model.load_state_dict(state_dict=chkpt["model_state_dict"])
    model.eval()

    # Load the games
    pgns = [os.path.join(GAMES_DIR, file) for file in os.listdir(GAMES_DIR)]

    # Read the boards from the games
    boards = []
    for pgn in pgns:
        boards += read_boards_from_pgn(pgn_file=pgn, start_move=end_opening, end_move=end_midgame)

    # Benchmark the beam search algorithm
    greedy_scores, mean_scores = beam_benchmark_boards(boards=boards[:100], model=model, max_depth=max_depth, max_width=max_width, max_workers=8)

    greedy_scores = pd.DataFrame(greedy_scores,
                                 index=[f"depth_{i}" for i in range(1, max_depth*2, 2)],
                                 columns=[f"width_{i}" for i in range(1, max_width + 1)])

    mean_scores = pd.DataFrame(mean_scores,
                               index=[f"depth_{i}" for i in range(1, max_depth*2, 2)],
                               columns=[f"width_{i}" for i in range(1, max_width + 1)])

    # save the results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    greedy_scores.to_csv(os.path.join(RESULTS_DIR, f"greedy_scores_{max_depth}_{max_width}.csv"))
    mean_scores.to_csv(os.path.join(RESULTS_DIR, f"mean_scores_{max_depth}_{max_width}.csv"))
