import os
import subprocess

from huggingface_hub import HfApi

space_local_folder = "./hf_space"
space_hf_id = "dechantoine/explainable-chess-engine"

if __name__ == "__main__":
    api = HfApi()

    # export poetry requirements
    command = [
        "poetry",
        "export",
        "-f",
        "requirements.txt",
        "--only",
        "hf-space",
        "--output",
        os.path.join(space_local_folder, "requirements.txt"),
    ]
    subprocess.run(command, check=True)

    # upload everything in the space folder
    api.upload_folder(
        folder_path=space_local_folder,
        repo_id=space_hf_id,
        repo_type="space",
        path_in_repo="",
    )
