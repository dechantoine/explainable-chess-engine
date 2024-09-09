import os

import cairosvg
import chess.svg
import graphviz
from anytree import AnyNode
from loguru import logger

graph_params = {
    "name": "Beam Search",
    "format": "png",
    "shape": "none",
    "board_size": 300,
    "fontsize": "20",
    "image_pos": "tc",
    "labelloc": "b",
    "labelfontsize": "20",
    "headport": "n",
    "tailport": "s",
}


@logger.catch(level="DEBUG")
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


@logger.catch(level="DEBUG")
def save_svg(
    board: chess.Board, filename: str, is_pruned: bool = False, to_png=False
) -> None:
    """Save a chess.Board object to an image file.

    Args:
        board (chess.Board): chess.Board object
        filename (str): filename to save the SVG file
        is_pruned (bool): whether the board is pruned or not
        to_png (bool): whether to save the SVG file as a PNG file

    """
    xml_declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    with open(filename + ".svg", "w") as f:
        f.write(
            xml_declaration
            + board_to_svg(board, is_pruned=is_pruned, size=graph_params["board_size"])
        )

    if to_png:
        cairosvg.svg2png(url=filename + ".svg", write_to=filename + ".png")


@logger.catch(level="DEBUG", reraise=True)
def plot_save_beam_search(
    beam: AnyNode,
    filename: str,
    temp_dir: str = "../reports/figures",
    intermediate_png: bool = False,
) -> None:
    """Plot and save the beam search tree.

    Args:
        beam (AnyNode): beam search tree
        filename (str): filename to save the tree plot
        temp_dir (str): temporary directory to save the SVG files
        intermediate_png (bool): whether to save the SVG files as PNG files

    """
    image_path = os.path.abspath(os.path.join(temp_dir, "root"))
    # logger.info(image_path)
    save_svg(board=beam.board, filename=image_path, to_png=intermediate_png)

    dot = graphviz.Digraph(name=graph_params["name"], format=graph_params["format"])

    image_path = image_path + ".png" if intermediate_png else image_path + ".svg"
    # logger.info(image_path)
    dot.node(
        name="ROOT",
        label="",
        labelloc=graph_params["labelloc"],
        fontsize=graph_params["fontsize"],
        shape=graph_params["shape"],
        image_pos=graph_params["image_pos"],
        image=image_path,
    )

    for board in beam.descendants:
        # a board is pruned if none of its children are at the same depth as the beam height
        # and if it is not at the beam height itself
        # and if it is not a terminal board itself or its children
        is_pruned = (
                not (any(board.depth == beam.height for board in board.descendants))
                and not (any(board.board.outcome() for board in board.descendants))
                and not (board.depth == beam.height)
                and not (board.board.outcome()))

        image_path = os.path.abspath(os.path.join(temp_dir, board.name))
        # logger.info(image_path)
        save_svg(
            board=board.board,
            filename=image_path,
            is_pruned=is_pruned,
            to_png=intermediate_png,
        )

        image_path = image_path + ".png" if intermediate_png else image_path + ".svg"
        # logger.info(image_path)
        dot.node(
            name=board.name,
            label="score = {:.4f}".format(board.score),
            labelloc=graph_params["labelloc"],
            fontsize=graph_params["fontsize"],
            shape=graph_params["shape"],
            image_pos=graph_params["image_pos"],
            image=image_path,
        )

        dot.edge(
            tail_name=board.parent.name,
            head_name=board.name,
            headlabel=board.move.uci(),
            fontsize=graph_params["labelfontsize"],
            headport=graph_params["headport"],
            tailport=graph_params["tailport"],
        )

    dot.render(
        filename=filename,
        format=graph_params["format"],
        # renderer="cairo",
        # formatter="cairo",
    )

    #logger.info(os.listdir(temp_dir))

    # log render file
    #with open(filename, "r") as f:
    #    logger.info(f.read())

    # delete all svg files
    # for file in os.listdir(temp_dir):
    #    if file.endswith(".svg"):
    #        logger.info("Deleting : " + os.path.join(temp_dir, file))
    #        os.remove(os.path.join(temp_dir, file))
