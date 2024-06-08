import os

import chess.svg
import graphviz
from anytree import AnyNode

graph_params = {
    "name": "Beam Search",
    "format": "png",
    "shape": "none",
    "fontsize": "20",
    "image_pos": "tc",
    "labelloc": "b",
    "labelfontsize": "20",
    "headport": "n",
    "tailport": "s",
}


def board_to_svg(board: chess.Board, is_pruned: bool = False, size: int = 360) -> str:
    """Convert a chess.Board object to an SVG string.

    Args:
        board (chess.Board): chess.Board object
        is_pruned (bool): whether the board is pruned or not
        size (int): size of the board

    Returns:
        str: SVG string of the board

    """
    svg = chess.svg.board(
        board=board,
        size=size,
        lastmove=board.peek() if board.move_stack else None,
        check=board.king(board.turn) if board.is_check() else None,
    )

    if is_pruned:
        svg = svg.replace("#d18b47", "#989898")
        svg = svg.replace("#ffce9e", "#d7d7d7")
        svg = svg.replace("#cdd16a", "#c4c4c4")
        svg = svg.replace("#aaa23b", "#999999")

    return svg


def save_svg(board: chess.Board, filename: str, is_pruned: bool = False):
    """Save a chess.Board object to an SVG file.

    Args:
        board (chess.Board): chess.Board object
        filename (str): filename to save the SVG file
        is_pruned (bool): whether the board is pruned or not

    """
    xml_declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    with open(filename, "w") as f:
        f.write(xml_declaration + board_to_svg(board, is_pruned=is_pruned))


def plot_save_beam_search(
    beam: AnyNode, filename: str, temp_dir: str = "../reports/figures"
) -> None:
    """Plot and save the beam search tree.

    Args:
        beam (AnyNode): beam search tree
        filename (str): filename to save the tree plot
        temp_dir (str): temporary directory to save the SVG files

    """
    save_svg(board=beam.board, filename=os.path.join(temp_dir, "root.svg"))

    dot = graphviz.Digraph(name=graph_params["name"], format=graph_params["format"])
    dot.node(
        name="ROOT",
        label="",
        labelloc=graph_params["labelloc"],
        fontsize=graph_params["fontsize"],
        shape=graph_params["shape"],
        image_pos=graph_params["image_pos"],
        image=os.path.join(temp_dir, "root.svg"),
    )

    for board in beam.descendants:
        # a board is pruned if none of its children are at the same depth as the beam height and if it is not at the
        # beam height
        is_pruned = not (
            any(board.depth == beam.height for board in board.descendants)
        ) and not (board.depth == beam.height)
        save_svg(
            board=board.board,
            filename=os.path.join(temp_dir, f"{board.name}.svg"),
            is_pruned=is_pruned,
        )

        dot.node(
            name=board.name,
            label="score = {:.4f}".format(board.score),
            labelloc=graph_params["labelloc"],
            fontsize=graph_params["fontsize"],
            shape=graph_params["shape"],
            image_pos=graph_params["image_pos"],
            image=os.path.join(temp_dir, f"{board.name}.svg"),
        )

        dot.edge(
            tail_name=board.parent.name,
            head_name=board.name,
            headlabel=board.move.uci(),
            fontsize=graph_params["labelfontsize"],
            headport=graph_params["headport"],
            tailport=graph_params["tailport"],
        )

    dot.render(filename=filename, format="png", cleanup=True)

    # delete all svg files
    for file in os.listdir(temp_dir):
        if file.endswith(".svg"):
            os.remove(os.path.join(temp_dir, file))
