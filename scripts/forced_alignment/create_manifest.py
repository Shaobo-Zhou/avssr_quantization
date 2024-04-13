import argparse
import os
from pathlib import Path

from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def create_alignment(dataset_name, audio_path, text_path, use_asr):
    dataset_path = ROOT_PATH / "data" / dataset_name
    audio_path = dataset_path / audio_path
    text_path = dataset_path / text_path

    output_dir = dataset_path / "alignment"
    output_dir.mkdir(exist_ok=True, parents=True)

    manifest_data = []

    for audio_part in ["train", "val", "test"]:
        if not (audio_path / audio_part).exists():
            continue

        wav_list = os.listdir(str(audio_path / audio_part))
        for wav_file in tqdm(wav_list):
            if not wav_file.endswith(".wav"):
                continue

            wav_path = audio_path / audio_part / wav_file
            filename = wav_file.split(".")[0]

            if use_asr:
                manifest_data.append({"audio_filepath": str(wav_path)})
            else:
                text_file = text_path / f"{filename}.txt"
                with open(text_file, "r") as f:
                    text = f.readline().strip()

                manifest_data.append({"audio_filepath": str(wav_path), "text": text})

    with (output_dir / "manifest.json").open("wt") as handle:
        for data in manifest_data:
            if use_asr:
                audio_filepath = data["audio_filepath"]
                audio_line = f'"audio_filepath": "{audio_filepath}"'
                handle.write("{" + audio_line + "}\n")
            else:
                audio_filepath = data["audio_filepath"]
                text = data["text"]
                audio_line = f'"audio_filepath": "{audio_filepath}"'
                text_line = f'"text": "{text}"'
                handle.write("{" + audio_line + ", " + text_line + "}\n")


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
        "-a",
        "--audio_path",
        default="audio",
        type=str,
        help="Path to audio within the dataset dir (default: audio)",
    )

    args.add_argument(
        "-t",
        "--text_path",
        default="text",
        type=str,
        help="Path to text within the dataset dir (default: text)",
    )

    args.add_argument(
        "-u",
        "--use_asr",
        default=False,
        type=bool,
        help="Use ASR prediction instead of ground truth text (default: False)",
    )

    args = args.parse_args()

    create_alignment(args.dataset_name, args.audio_path, args.text_path, args.use_asr)
