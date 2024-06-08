import os

import chess.pgn
import chess.svg
import gradio as gr
import torch
from PIL import Image

from demos.temp.demo_utils import clean_board
from src.agents.policies import beam_search, eval_board
from src.agents.viz_utils import plot_save_beam_search
from src.models.simple_feed_forward import SimpleFF

TEMP_DIR = "./demos/temp/"

chkpt = torch.load(f="./models_checkpoint/simple_ff_0/checkpoint_36700.pt")
model = SimpleFF()
model.load_state_dict(state_dict=chkpt["model_state_dict"])
model.eval()

os.makedirs(name=TEMP_DIR, exist_ok=True)


def evaluate_board(board: chess.Board):
    """Evaluate the board.

    Args:
        board (chess.Board): chess.Board object

    Returns:
        float: score of the board

    """
    board = clean_board(board=board)
    svg = chess.svg.board(
        board=board,
        size=360,
        lastmove=board.peek() if board.move_stack else None,
        check=board.king(board.turn) if board.is_check() else None,
    )
    xml_declaration = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    with open(os.path.join(TEMP_DIR, "board.svg"), "w") as f:
        f.write(xml_declaration + svg)

    return os.path.join(TEMP_DIR, "board.svg"), eval_board(model=model, board=board)


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
        filename=os.path.abspath(os.path.join(TEMP_DIR, "beam_search")),
        temp_dir=os.path.abspath(TEMP_DIR),
    )

    return Image.open(fp=os.path.join(TEMP_DIR, "beam_search.png"))


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
                gr.Slider(value=4, minimum=1, maximum=10, step=1),
                gr.Slider(value=4, minimum=1, maximum=10, step=1),
            ],
            outputs="image",
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
        )

demo.launch()
