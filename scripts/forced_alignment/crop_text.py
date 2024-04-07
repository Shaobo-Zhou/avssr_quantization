import argparse
import os
from pathlib import Path

from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def parse_ctm(ctm_file, seconds):
    with open(ctm_file, "r") as f:
        lines = f.readlines()

    output_text = ""

    for line in lines:
        line_split = line.split(" ")
        start_time = float(line_split[2])
        token = line_split[4]
        if token[0] == "<":
            token = " "
        if token[0] == "▁":
            token = " " + token[1:]
        if start_time < seconds:
            output_text = output_text + token

    return output_text.strip()


def crop_text(dataset_name, seconds):
    dataset_path = ROOT_PATH / "data" / dataset_name

    alignment_path = dataset_path / "alignment" / "nfa_output"
    tokens_path = alignment_path / "ctm" / "tokens"

    output_dir = dataset_path / "cropped_text"
    output_dir.mkdir(exist_ok=True, parents=True)

    ctm_list = os.listdir(str(tokens_path))
    for ctm_file in tqdm(ctm_list):
        if not ctm_file.endswith(".ctm"):
            continue

        text = parse_ctm(str(tokens_path / ctm_file), seconds)

        filename = ctm_file.split(".")[0]

        text_file = output_dir / f"{filename}.txt"
        with open(text_file, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Create Manifest for Alignment")
    args.add_argument(
        "-d",
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name inside data dir (default: None)",
    )

    args.add_argument(
        "-s",
        "--seconds",
        default=2.0,
        type=float,
        help="Number of seconds to crop (default: 2.0)",
    )

    args = args.parse_args()

    crop_text(args.dataset_name, args.seconds)
