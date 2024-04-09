import os
import re

import numpy as np
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class AVDataset(BaseDataset):
    def __init__(
        self,
        dataset_name,
        audio_path,
        video_path,
        text_path,
        name="tr",
        n_src=1,
        *args,
        **kwargs,
    ):
        index_path = (
            ROOT_PATH / "data" / dataset_name / f"{name}_n_src={n_src}_index.json"
        )

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(
                dataset_name, audio_path, video_path, text_path, name, n_src
            )

        super().__init__(index, n_src=n_src, *args, **kwargs)

    def _create_index(
        self, dataset_name, audio_path, video_path, text_path, name, n_src
    ):
        index = []
        data_path = ROOT_PATH / "data" / dataset_name
        embedding_path = ROOT_PATH / "data" / "embeddings" / dataset_name
        audio_path = data_path / audio_path / name
        video_path = data_path / video_path
        text_path = data_path / text_path

        print(f"Creating AVDataset Index for part {name}")

        wav_list = os.listdir(str(audio_path / "mix"))

        for wav_file in tqdm(wav_list):
            if not wav_file.endswith(".wav"):
                continue
            mix_path = audio_path / "mix" / wav_file
            s1_path = audio_path / "s1" / wav_file
            s2_path = audio_path / "s2" / wav_file
            wav_file_split = wav_file.split("_")

            mix_id = wav_file[:-4]  # remove .wav
            s1_id = f"{wav_file_split[0]}_{wav_file_split[1]}"
            s2_id = f"{wav_file_split[3]}_{wav_file_split[4]}"
            mix_s1_id = f"{mix_id}|{s1_id}"
            mix_s2_id = f"{mix_id}|{s2_id}"

            embedding_s1_path = embedding_path / f"{mix_s1_id}.pt"
            embedding_s2_path = embedding_path / f"{mix_s2_id}.pt"

            s1_video_path = video_path / f"{s1_id}.npz"
            s2_video_path = video_path / f"{s2_id}.npz"

            s1_text_path = text_path / f"{s1_id}.txt"
            s2_text_path = text_path / f"{s2_id}.txt"

            with open(s1_text_path, "r") as f:
                s1_text = f.readline().strip()
                s1_text = normalize_text(s1_text)

            with open(s2_text_path, "r") as f:
                s2_text = f.readline().strip()
                s2_text = normalize_text(s2_text)

            t_info = torchaudio.info(str(mix_path))
            audio_length = t_info.num_frames

            if n_src == 2:
                index.append(
                    {
                        "mix_path": str(mix_path),
                        "s1_path": str(s1_path),
                        "s2_path": str(s2_path),
                        "s1_video_path": str(s1_video_path),
                        "s2_video_path": str(s2_video_path),
                        "s1_text": s1_text,
                        "s2_text": s2_text,
                        "s1_id": s1_id,
                        "s2_id": s2_id,
                        "mix_id": mix_id,
                        "mix_s1_id": mix_s1_id,
                        "mix_s2_id": mix_s2_id,
                        "embedding_s1_path": str(embedding_s1_path),
                        "embedding_s2_path": str(embedding_s2_path),
                        "audio_length": audio_length,
                    }
                )
            if n_src == 1:  # s1 and s2 are two elements of dataset
                index.append(
                    {
                        "mix_path": str(mix_path),
                        "s_path": str(s1_path),
                        "s_video_path": str(s1_video_path),
                        "s_text": s1_text,
                        "s_id": s1_id,
                        "mix_id": mix_id,
                        "mix_s_id": mix_s1_id,
                        "embedding_s_path": str(embedding_s1_path),
                        "audio_length": audio_length,
                    }
                )
                index.append(
                    {
                        "mix_path": str(mix_path),
                        "s_path": str(s2_path),
                        "s_video_path": str(s2_video_path),
                        "s_text": s2_text,
                        "s_id": s2_id,
                        "mix_id": mix_id,
                        "mix_s_id": mix_s2_id,
                        "embedding_s_path": str(embedding_s2_path),
                        "audio_length": audio_length,
                    }
                )

        write_json(index, str(data_path / f"{name}_n_src={n_src}_index.json"))

        return index


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text
