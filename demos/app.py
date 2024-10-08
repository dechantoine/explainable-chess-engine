import os
import sys

import chess.pgn
import gradio as gr
import numpy as np
import torch
from loguru import logger

from src.data.data_utils import clean_board
from src.engine.agents.dl_agent import choose_move_from_beam
from src.engine.agents.policies import beam_search, eval_board, one_depth_eval
from src.engine.agents.viz_utils import plot_save_beam_search, save_svg
from src.models.model_space import MultiInputConv

# TEMP_DIR = "./demos/temp/"
CHKPT = "checkpoint.pt"

file = sys.argv[0]
DIR_PATH = os.path.dirname(file)
TEMP_DIR = os.path.join(DIR_PATH, "temp")

chkpt = torch.load(os.path.join(DIR_PATH, f"checkpoints/{CHKPT}"))
model = MultiInputConv()
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
def plot_beam_search(board: chess.Board,
                     depth: int,
                     beam_width: int,
                     player_strategy: str,
                     opponent_strategy: str,
                     player_top_k: int,
                     opponent_top_k: int):
    """Plot the beam search tree.

    Args:
        board (chess.Board): chess.Board object
        depth (int): depth of the search
        beam_width (int): width of the beam
        player_strategy (str): sampling strategy
        opponent_strategy (str): sampling strategy
        player_top_k (int): top-k value
        opponent_top_k (int): top-k value

    Returns:
        Image: image of the beam search tree
        str: next move according to policy

    """
    board = clean_board(board=board)

    beam = beam_search(model=model,
                       board=board,
                       depth=depth,
                       beam_width=beam_width,
                       player_strategy=player_strategy,
                       opponent_strategy=opponent_strategy,
                       player_top_k=player_top_k,
                       opponent_top_k=opponent_top_k,
                       min_score=-100,
                       max_score=100)

    next_move = choose_move_from_beam(beam=beam, is_white=board.turn, gamma=0.9)

    plot_save_beam_search(
        beam=beam,
        filename=os.path.join(TEMP_DIR, "beam_search"),
        temp_dir=TEMP_DIR,
        intermediate_png=False,
    )

    return os.path.join(TEMP_DIR, "beam_search.png"), str(next_move)


@logger.catch(level="DEBUG", reraise=True)
def get_one_depth_eval(board: chess.Board):
    """Get the legal boards from one-depth evaluation.

    Args:
        board (chess.Board): chess.Board object

    Returns:
        list: list of tuples of SVG images and scores of the legal boards

    """
    board = clean_board(board=board)

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

    [save_svg(board=board, filename=os.path.join(TEMP_DIR, f"board_{i}"), to_png=False) for i, board in
     enumerate(legal_boards)]

    return (gr.update(value=[(os.path.join(TEMP_DIR, f"board_{i}.svg"), str(scores[i])) for i in range(len(legal_boards))]),
            gr.update(choices=[str(move) for move in legal_moves]),
            [str(move) for move in legal_moves])

@logger.catch(level="DEBUG", reraise=True)
def select_dropdown_item(moves, evt: gr.SelectData):
    """Select the nth item in the dropdown.

    Args:
        moves (list): list of moves
        evt (gr.EventData): event data

    """
    selected_index = evt.index
    return gr.update(value=moves[selected_index])

@logger.catch(level="DEBUG", reraise=True)
def update_run_fen(fen, dropdown):
    """Update the FEN board with the selected move.

    Args:
        fen (str): FEN board
        dropdown (str): selected move

    """
    board = clean_board(board=fen)
    board.push_san(dropdown)

    boards, update_dropdown, moves = get_one_depth_eval(board.fen())

    return board.fen(), boards, update_dropdown, moves

def update_top_k_visibility(strategy):
    return gr.update(visible=(strategy == "top-k"))


with gr.Blocks() as demo:
    gr.Markdown("Explore the model")
    with gr.Tab("Beam search"):
        with gr.Row():

            with gr.Column():

                board = gr.Textbox(
                    value="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
                    label="Provide FEN or PGN board:",
                )
                depth_slider = gr.Slider(value=4, minimum=1, maximum=10, step=1, label="Choose beam depth")
                width_slider = gr.Slider(value=4, minimum=1, maximum=10, step=1, label="Choose beam width")

                player_strategy = gr.Dropdown(
                    label="Select the player sampling strategy :",
                    choices=["greedy", "top-k"],
                    value="greedy",
                    interactive=True,
                    allow_custom_value=False,
                )

                opponent_strategy = gr.Dropdown(
                    label="Select the opponent sampling strategy :",
                    choices=["greedy", "top-k"],
                    value="greedy",
                    interactive=True,
                    allow_custom_value=False,
                )

                player_top_k = gr.Slider(value=5,
                                  minimum=5,
                                  maximum=20,
                                  step=1,
                                  label="Choose player top-k",
                                  interactive=True,
                                  visible=False)

                opponent_top_k = gr.Slider(value=2,
                                           minimum=2,
                                           maximum=20,
                                           step=1,
                                           label="Choose opponent top-k",
                                           interactive=True,
                                           visible=False)

                btn = gr.Button("Run beam search")

            with gr.Column():
                beam = gr.Image()
                next_move = gr.Textbox(label="Next move according to current policy:", value="")

        player_strategy.change(fn=lambda x: gr.update(visible=(x == "top-k")), inputs=[player_strategy], outputs=[player_top_k])
        opponent_strategy.change(fn=lambda x: gr.update(visible=(x == "top-k")), inputs=[opponent_strategy], outputs=[opponent_top_k])
        width_slider.change(fn=lambda x: gr.update(minimum=x + 1, value=x + 1), inputs=[width_slider], outputs=[player_top_k])
        btn.click(fn=plot_beam_search,
                  inputs=[board, depth_slider, width_slider, player_strategy, opponent_strategy, player_top_k, opponent_top_k],
                  outputs=[beam, next_move])


    with gr.Tab("One-depth eval"):
        moves = gr.State(value=[])

        board = gr.Textbox(
            value="rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
            label="Provide FEN or PGN board:",
        )
        btn = gr.Button("Get one-depth evaluation")

        gallery = gr.Gallery(
            label="Legal boards from one-depth eval",
            show_label=False,
            elem_id="gallery",
            columns=6,
            interactive=False
        )

        dropdown = gr.Dropdown(
            label="Select the next move :",
            interactive=True,
        )

        btn_replace = gr.Button("Append selected move and run evaluation")

        btn.click(fn=get_one_depth_eval, inputs=[board], outputs=[gallery, dropdown, moves])
        gallery.select(fn=select_dropdown_item, inputs=[moves], outputs=dropdown)
        btn_replace.click(fn=update_run_fen, inputs=[board, dropdown], outputs=[board, gallery, dropdown, moves])

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
