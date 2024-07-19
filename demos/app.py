import os
import sys

import chess.pgn
import gradio as gr
import torch
from loguru import logger

from src.data.data_utils import clean_board
from src.engine.agents.policies import beam_search, eval_board
from src.engine.agents.viz_utils import plot_save_beam_search, save_svg
from src.models.simple_feed_forward import SimpleFF

# TEMP_DIR = "./demos/temp/"
CHKPT = "checkpoint_36700.pt"

file = sys.argv[0]
DIR_PATH = os.path.dirname(file)
TEMP_DIR = os.path.join(DIR_PATH, "temp")

chkpt = torch.load(os.path.join(DIR_PATH, f"models/{CHKPT}"))
model = SimpleFF()
model.load_state_dict(state_dict=chkpt["model_state_dict"])
model.eval()

os.makedirs(name=TEMP_DIR, exist_ok=True)


@logger.catch(level="DEBUG", reraise=True)
def evaluate_board(board: chess.Board):
    """Evaluate the board.

    Args:
        board (chess.Board): chess.Board object

    Returns:
        float: score of the board

    """
    board = clean_board(board=board)
    save_svg(board=board, filename=os.path.join(TEMP_DIR, "board"), to_png=False)

    return os.path.join(TEMP_DIR, "board.svg"), eval_board(model=model, board=board)


@logger.catch(level="DEBUG", reraise=True)
def plot_beam_search(board: chess.Board, depth: int, beam_width: int):
    """Plot the beam search tree.

    Args:
        board (chess.Board): chess.Board object
        depth (int): depth of the search
        beam_width (int): width of the beam

    Returns:
        Image: image of the beam search tree

    """
    board = clean_board(board=board)

    beam = beam_search(model=model, board=board, depth=depth, beam_width=beam_width)
    plot_save_beam_search(
        beam=beam,
        filename=os.path.join(TEMP_DIR, "beam_search"),
        temp_dir=TEMP_DIR,
        intermediate_png=True,
    )

    return os.path.join(TEMP_DIR, "beam_search.png")


with gr.Blocks() as demo:
    gr.Markdown("Explore the model")
    with gr.Tab("Beam search"):
        gr.Interface(
            fn=plot_beam_search,
            inputs=[
                gr.Textbox(
                    value="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
                    label="Provide FEN or PGN board:",
                ),
                gr.Slider(value=4, minimum=1, maximum=10, step=1, label="Depth"),
                gr.Slider(value=4, minimum=1, maximum=10, step=1, label="Beam width"),
            ],
            outputs="image",
            allow_flagging="never",
        )

    with gr.Tab("Score a board"):
        gr.Interface(
            fn=evaluate_board,
            inputs=[
                gr.Textbox(
                    value="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
                    label="Provide FEN or PGN board:",
                ),
            ],
            outputs=["image", "text"],
            allow_flagging="never",
        )


if __name__ == "__main__":
    demo.launch()
