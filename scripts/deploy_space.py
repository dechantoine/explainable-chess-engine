import os
import shutil
import subprocess

from huggingface_hub import HfApi

demo_local_folder = "./demos"
space_local_folder = "./hf_space"
space_hf_id = "dechantoine/explainable-chess-engine"

if __name__ == "__main__":
    api = HfApi()

    # create the space folder
    try:
        os.makedirs(space_local_folder)
    except FileExistsError:
        shutil.rmtree(space_local_folder)
        os.makedirs(space_local_folder)

    # copy all the files from the demo folder to the space folder
    command = ["cp", "-r", f"{demo_local_folder}/.", space_local_folder]
    subprocess.run(command, check=True)

    # copy all the files from the src folder to the space folder
    os.makedirs(f"{space_local_folder}/src/engine/agents", exist_ok=True)
    command = ["cp", "-r", "./src/engine/agents/.", f"{space_local_folder}/src/engine/agents"]
    subprocess.run(command, check=True)

    # copy src/data/data_utils.py to the space folder
    os.makedirs(f"{space_local_folder}/src/data", exist_ok=True)
    command = ["cp", "./src/data/data_utils.py", f"{space_local_folder}/src/data"]
    subprocess.run(command, check=True)

    # copy src/models/model_space to the space folder
    os.makedirs(f"{space_local_folder}/src/models", exist_ok=True)
    command = ["cp", "./src/models/model_space.py", f"{space_local_folder}/src/models"]
    subprocess.run(command, check=True)

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

    # delete the space folder
    # command = ["rm", "-r", space_local_folder]
    # subprocess.run(command, check=True)
