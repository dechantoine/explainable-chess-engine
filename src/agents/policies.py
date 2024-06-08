import chess.pgn
import more_itertools
import numpy as np
import torch
from anytree import AnyNode, LevelOrderGroupIter

from src.data.data_utils import batch_boards_to_tensor


def eval_board(model: torch.nn.Module, board: chess.Board) -> float:
    """Evaluate a single board.

    Args:
        model (torch.nn.Module): model to evaluate the board
        board (chess.Board): board to evaluate

    Returns:
        float: score of the board

    """
    return model(torch.from_numpy(batch_boards_to_tensor([board]))).item()


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


def one_depth_eval(model: torch.nn.Module, boards: list[chess.Board]) -> tuple:
    """Evaluate one depth of the tree.

    Args:
        model (torch.nn.Module): model to evaluate the boards
        boards (list): list of chess.Board objects

    Returns:
        tuple: tuple of legal boards, legal moves to get to the legal boards and scores of the legal boards

    """
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

    scores = (
        model(torch.from_numpy(batch_boards_to_tensor(legal_boards)))
        .detach()
        .numpy()
        .flatten()
    )

    # reshape the scores and boards to the original shape
    scores = list(more_itertools.split_into(scores, legal_boards_shape))
    legal_boards = list(more_itertools.split_into(legal_boards, legal_boards_shape))

    return legal_boards, legal_moves, scores


def beam_search(
    model: torch.nn.Module, board: chess.Board, depth: int = 2, beam_width: int = 5
) -> AnyNode:
    """Beam search algorithm to evaluate the next moves. The algorithm maximizes (minimizes) the score of the board for
    white (black) player. It assumes that the opponent plays the best greedy move.

    Args:
        model (torch.nn.Module): model to evaluate the boards
        board (chess.Board): board to start the search from
        depth (int): depth of the search
        beam_width (int): width of the beam

    Returns:
        AnyNode: tree of the best moves

    """
    root = AnyNode(name="ROOT", board=board.copy(), score=None, move=None)

    is_white = board.turn
    is_opponent = False

    for d in range(depth):
        best_nodes = list(LevelOrderGroupIter(root))[-1]
        best_boards = [node.board for node in best_nodes]

        boards, moves, scores = one_depth_eval(model=model, boards=best_boards)

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

            # get the beam_width best scores
            level_width = min(beam_width, len(scores))
            if is_white:
                partition = np.arange(-1, -(level_width + 1), -1)
            else:
                partition = np.arange(level_width)

            idx = np.argpartition(scores, partition)[partition]

            # add the best boards, scores and moves to the tree
            for i in range(len(idx)):
                # find the parent of the node by inserting its index in the cumulative sum of children count
                parent = np.searchsorted(np.cumsum(children_count), idx[i])
                AnyNode(
                    name=f"Depth {d} Candidate {i}",
                    parent=best_nodes[parent],
                    board=boards[idx[i]],
                    score=scores[idx[i]],
                    move=moves[idx[i]],
                )

        if is_opponent:
            if is_white:
                idx = [np.argmax(scores[i]) for i in range(len(scores))]
            else:
                idx = [np.argmin(scores[i]) for i in range(len(scores))]

            # add the best boards, scores and moves to the tree
            for i in range(len(idx)):
                AnyNode(
                    name=f"Depth {d} Candidate {i}",
                    parent=best_nodes[i],
                    board=boards[i][idx[i]],
                    score=scores[i][idx[i]],
                    move=moves[i][idx[i]],
                )

        is_white = not is_white
        is_opponent = not is_opponent

    return root
