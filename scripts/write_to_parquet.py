import chess.pgn
import click
from loguru import logger
from stockfish import Stockfish
from tqdm import tqdm

from src.data.parquet_db import ParquetChessDB


def init_stockfish():
    return Stockfish(
        depth=10,
        parameters=
        {
            "Debug Log File": "",
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 16,  # The number of CPU threads used for searching a position.
            "Ponder": "false",  # Let Stockfish ponder its next move while the opponent is thinking.
            "Hash": 4096,  # The size of the hash table in MB.
            "MultiPV": 1,  # Output the N best lines (principal variations, PVs) when searching.
            "Skill Level": 20,
            # Lower the Skill Level in order to make Stockfish play weaker (see also UCI_LimitStrength).
            # "Move Overhead": 10,  # Assume a time delay of x ms due to network and GUI overheads.
            # "Minimum Thinking Time": 20,
            # "Slow Mover": 100,
            # "UCI_Chess960": "false",
            # "UCI_LimitStrength": "true",  # Enable weaker play aiming for an Elo rating as set by UCI_Elo
            # "UCI_Elo": 3000  # If UCI_LimitStrength is enabled, it aims for an engine strength of the given Elo.
        }
    )


@logger.catch
def evaluate_boards(boards: list[chess.Board]) -> list[float]:
    stockfish = init_stockfish()
    values = []
    for b in tqdm(iterable=boards,
                  desc="Evaluating boards with Stockfish...",):
        stockfish.set_fen_position(b.fen())
        stockfish_eval = stockfish.get_evaluation()
        if stockfish_eval["type"] == "cp":
            v = stockfish_eval["value"] / 100
        else:
            v = -100 if stockfish_eval["value"] < 0 else 100
        values.append(v)
    return values


@click.group()
def cli():
    pass


@click.command()
@click.option("--input_path", required=True, help="Directory containing PGN files.")
@click.option("--output_path", required=True, help="Output directory for Parquet dataset.")
@click.option("--stockfish_eval", default=True, help="Whether to evaluate boards with Stockfish.")
def process_directory(input_path: str, output_path: str, stockfish_eval: bool = True) -> None:
    logger.info(f"Processing directory {input_path} and writing to {output_path}")

    db = ParquetChessDB(output_path)
    logger.info("ParquetChessDB initialized.")

    funcs = None
    if stockfish_eval:
        funcs = {"stockfish_eval": evaluate_boards}

    db.add_directory(directory=input_path,
                     funcs=funcs
                     )


if __name__ == "__main__":
    cli.add_command(process_directory)
    cli()
