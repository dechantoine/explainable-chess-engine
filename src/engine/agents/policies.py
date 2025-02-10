from enum import Enum

import chess.pgn
import more_itertools
import numpy as np
import torch
from anytree import AnyNode, LevelOrderGroupIter

from ...data.data_utils import batch_boards_to_tensor


class Strategy(Enum):
    GREEDY = "greedy"
    TOP_K = "top-k"


def eval_board(model: torch.nn.Module, board: chess.Board) -> float:
    """Evaluate a single board.

    Args:
        model (torch.nn.Module): model to evaluate the board.
        board (chess.Board): board to evaluate.

    Returns:
        float: score of the board

    """
    tensors = batch_boards_to_tensor([board])
    return float(model(tensors).detach().numpy().flatten()[0])


def get_legal_moves(boards: list[chess.Board]) -> list[list[chess.Move]]:
    """Get legal moves for a batch of boards.

    Args:
        boards (list): list of chess.Board objects

    Returns:
        list: list of legal moves for each board

    """
    return [list(board.generate_legal_moves()) for board in boards]


def push_legal_moves(
        boards: list[chess.Board], legal_moves: list[list[chess.Move]]
) -> list[list[chess.Board]]:
    """Push legal moves for a batch of boards.

    Args:
        boards (list): list of chess.Board objects
        legal_moves (list): list of legal moves for each board

    Returns:
        list: list of shape (len(boards), len(legal_moves)) of boards after pushing legal moves

    """
    pushed_boards = [
        [board.copy() for _ in range(len(legal_moves[i]))]
        for i, board in enumerate(boards)
    ]
    [
        [
            pushed_boards[i][j].push(legal_moves[i][j])
            for j in range(len(legal_moves[i]))
        ]
        for i in range(len(boards))
    ]
    return pushed_boards


def one_depth_eval(model: torch.nn.Module,
                   boards: list[chess.Board],
                   min_score: float = -1.0,
                   max_score: float = 1.0
                   ) -> tuple:
    """Evaluate one depth of the tree.

    Args:
        model (torch.nn.Module): model to evaluate the boards
        boards (list): list of chess.Board objects
        min_score (float): minimum score of the model
        max_score (float): maximum score of the model

    Returns:
        tuple: tuple of legal boards, legal moves to get to the legal boards and scores of the legal boards

    """
    # get legal boards for each legal move for each board
    legal_moves = get_legal_moves(boards)
    legal_boards = push_legal_moves(boards, legal_moves)

    # keep the shape of the legal boards
    legal_boards_shape = [len(legal_boards[i]) for i in range(len(legal_boards))]

    # flatten the list of boards
    legal_boards = [
        legal_boards[i][j]
        for i in range(len(legal_boards))
        for j in range(len(legal_boards[i]))
    ]

    # check if some boards are in the terminal state
    terminal_boards = {i: board.outcome() for i, board in enumerate(legal_boards) if board.outcome()}

    scores = (
        model(batch_boards_to_tensor(legal_boards))
        .detach()
        .numpy()
        .flatten()
    )

    for i, reason in terminal_boards.items():
        if reason.result() == "1-0":
            scores[i] = max_score
        elif reason.result() == "0-1":
            scores[i] = min_score
        else:
            scores[i] = 0.0

    # reshape the scores to the original shape
    scores = list(more_itertools.split_into(scores, legal_boards_shape))

    # reshape the legal boards to the original shape
    legal_boards = list(more_itertools.split_into(legal_boards, legal_boards_shape))

    return legal_boards, legal_moves, scores


def beam_sampling(boards: list[list[chess.Board]],
                  scores: list[list[np.float32]],
                  moves: list[list[chess.Move]],
                  beam_width: int,
                  strategy: str = "greedy",
                  top_k: int = 5,
                  is_white: bool = True,
                  is_opponent: bool = False) -> list[dict]:
    """Sample the beam_width best boards.

    Args:
        boards (list): list of chess.Board objects for each node
        scores (list): list of scores of the boards
        moves (list): list of moves to get to the boards
        beam_width (int): width of the beam
        strategy (str): strategy to sample the best boards
        top_k (int): top k to sample the best boards
        is_white (bool): whether the player is white
        is_opponent (bool): whether the player is the opponent

    Returns:
        tuple: tuple of the beam_width best boards, scores and moves

    """
    try:
        strategy = Strategy(strategy)
    except ValueError:
        raise ValueError("Invalid strategy. Must be one of 'greedy', 'top-cumsum' or 'top-k'.")

    if not is_opponent:
        # save the children count of each node
        children_count = [len(boards[i]) for i in range(len(boards))]

        # flatten the list of scores, boards and moves
        scores = [
            scores[i][j] for i in range(len(scores)) for j in range(len(scores[i]))
        ]
        boards = [
            boards[i][j] for i in range(len(boards)) for j in range(len(boards[i]))
        ]
        moves = [
            moves[i][j] for i in range(len(moves)) for j in range(len(moves[i]))
        ]

        level_width = min(beam_width, len(scores))

        if strategy == Strategy.GREEDY:
            # get the beam_width best scores
            if is_white:
                partition = np.arange(-1, -(level_width + 1), -1)
            else:
                partition = np.arange(level_width)

            idx = np.argpartition(scores, partition)[partition]

        elif strategy == Strategy.TOP_K:
            # get the top_k best scores
            if is_white:
                idx_candidates = np.argsort(scores)[::-1][:max(top_k, level_width)]
            else:
                idx_candidates = np.argsort(scores)[:max(top_k, level_width)]

            idx = np.random.choice(idx_candidates, size=level_width, replace=False)

        return [{"candidate_id": i,
                 "parent_id": np.searchsorted(np.cumsum(children_count), idx[i]),
                 "board": boards[idx[i]],
                 "score": scores[idx[i]],
                 "move": moves[idx[i]]} for i in range(len(idx))]

    if is_opponent:

        if strategy == Strategy.GREEDY:
            if is_white:
                idx = [np.argmax(scores[i]) for i in range(len(scores))]
            else:
                idx = [np.argmin(scores[i]) for i in range(len(scores))]

        elif strategy == Strategy.TOP_K:
            if is_white:
                idx = [np.random.choice(np.argsort(scores[i])[::-1][:top_k], size=1)[0] for i in range(len(scores))]
            else:
                idx = [np.random.choice(np.argsort(scores[i])[:top_k], size=1)[0] for i in range(len(scores))]

        return [{"candidate_id": i,
                 "parent_id": i,
                 "board": boards[i][idx[i]],
                 "score": scores[i][idx[i]],
                 "move": moves[i][idx[i]]} for i in range(len(idx))]


def beam_search(
        model: torch.nn.Module,
        board: chess.Board,
        depth: int = 2,
        beam_width: int = 5,
        player_strategy: str = "greedy",
        opponent_strategy: str = "greedy",
        player_top_k: int = 5,
        opponent_top_k: int = 5,
        min_score: float = -1.0,
        max_score: float = 1.0
) -> AnyNode:
    """Beam search algorithm to evaluate the next moves. The algorithm maximizes (minimizes) the score of the board for
    white (black) player. It assumes that the opponent plays the best greedy move.

    Args:
        model (torch.nn.Module): model to evaluate the boards
        board (chess.Board): board to start the search from
        depth (int): depth of the search
        beam_width (int): width of the beam
        player_strategy (str): strategy to sample the best boards for the player
        opponent_strategy (str): strategy to sample the best boards for the opponent
        player_top_k (int): top k to sample the best boards for the player
        opponent_top_k (int): top k to sample the best boards for the opponent
        min_score (float): minimum score of the model
        max_score (float): maximum score of the model

    Returns:
        AnyNode: tree of the best moves

    """
    root = AnyNode(name="ROOT",
                   board=board.copy(),
                   score=None,
                   move=None,
                   # terminal=None
                   )

    is_white = board.turn
    is_opponent = False

    for d in range(depth):
        best_nodes = list(LevelOrderGroupIter(root))[-1]
        best_boards = [node.board for node in best_nodes]

        # extract terminal state boards
        terminal_boards = {i: board.outcome() for i, board in enumerate(best_boards) if board.outcome()}

        # remove terminal state boards and nodes
        best_boards = [board for i, board in enumerate(best_boards) if i not in terminal_boards.keys()]
        best_nodes = [node for i, node in enumerate(best_nodes) if i not in terminal_boards.keys()]

        # update the beam width to take into account the terminal boards
        beam_width -= len(terminal_boards)
        if beam_width <= 0 or len(best_boards) == 0:
            break

        boards, moves, scores = one_depth_eval(model=model, boards=best_boards, min_score=min_score,
                                               max_score=max_score)

        nodes = beam_sampling(boards=boards,
                              scores=scores,
                              moves=moves,
                              beam_width=beam_width,
                              strategy=player_strategy if not is_opponent else opponent_strategy,
                              top_k=player_top_k if not is_opponent else opponent_top_k,
                              is_white=is_white,
                              is_opponent=is_opponent)

        for n in nodes:
            AnyNode(
                name=f"Depth {d} Candidate {n['candidate_id']}",
                parent=best_nodes[n["parent_id"]],
                board=n["board"],
                score=float(n["score"]),
                move=n["move"],
                # outcome=n["board"].outcome()
            )

        is_white = not is_white
        is_opponent = not is_opponent

    return root
