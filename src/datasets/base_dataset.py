import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        n_src=1,
        min_audio_length=32000,
        limit=None,
        instance_transforms=None,
        text_encoder=None,
    ):
        assert n_src == 1, "Currently only one target source is supported, set n_src=1"
        self._assert_index_is_valid(index, n_src)
        assert (
            text_encoder is not None
        ), "For now you should always provide text_encoder"

        self.n_src = n_src

        index = self._filter_records_from_dataset(index, min_audio_length)
        index = self._shuffle_and_limit_index(index, limit)
        index = self._sort_index(index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

        self.text_encoder = text_encoder

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        # n_src == 1 is only supported for now
        mix_audio = self.load_audio(data_dict["mix_path"])
        s_audio = self.load_audio(data_dict["s_path"])
        s_video = self.load_video(data_dict["s_video_path"])
        s_text = data_dict["s_text"]
        s_tokens = self.text_encoder.encode(s_text)
        s_audio_length = data_dict["audio_length"]

        s_audio, s_video = self.process_data(s_audio, s_video)

        result_dict = {
            "mix_audio": mix_audio,
            "s_audio": s_audio,
            "s_video": s_video.unsqueeze(0),
            "s_text": s_text,
            "s_tokens": s_tokens,
            "s_tokens_length": s_tokens.shape[0],
            "s_audio_length": s_audio_length,
        }

        return result_dict

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio, sr = torchaudio.load(path)
        return audio

    def load_video(self, path):
        video = np.load(path)["data"]
        video = torch.tensor(video, dtype=torch.float32) / 255
        return video

    def process_data(self, s_audio, s_video):
        if self.instance_transforms is not None:
            if self.instance_transforms.get("video") is not None:
                s_video = self.instance_transforms.video(s_video)
            if self.instance_transforms.get("audio") is not None:
                s_audio = self.instance_transforms.audio(s_audio)
        return s_audio, s_video

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        min_audio_length,
    ) -> list:
        old_length = len(index)
        index = [entry for entry in index if entry["audio_length"] >= min_audio_length]
        new_length = len(index)

        filtered = new_length - old_length

        print(
            f"Filtered {filtered}({filtered / old_length:.1%}) records  from dataset that were smaller than {min_audio_length}"
        )

        return index

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_length"])

    @staticmethod
    def _assert_index_is_valid(index, n_src):
        for entry in index:
            assert "mix_path" in entry, (
                "Each dataset item should include field 'mix_path'"
                " - path to mix audio file."
            )
            if n_src == 1:
                assert "s_path" in entry, (
                    "Each dataset item should include field 's1_path'"
                    " - path to s1 audio file."
                )
                assert "s_video_path" in entry, (
                    "Each dataset item should include field 's1_video_path'"
                    " - path to s1 video file."
                )
                assert "s_text" in entry, (
                    "Each dataset item should include field 's1_text'"
                    " - str with s1 text."
                )
            if n_src == 2:
                assert "s1_path" in entry, (
                    "Each dataset item should include field 's1_path'"
                    " - path to s1 audio file."
                )
                assert "s1_video_path" in entry, (
                    "Each dataset item should include field 's1_video_path'"
                    " - path to s1 video file."
                )
                assert "s1_text" in entry, (
                    "Each dataset item should include field 's1_text'"
                    " - str with s1 text."
                )
                assert "s2_path" in entry, (
                    "Each dataset item should include field 's2_path'"
                    " - path to s2 audio file."
                )
                assert "s2_video_path" in entry, (
                    "Each dataset item should include field 's2_video_path'"
                    " - path to s2 video file."
                )

                assert "s2_text" in entry, (
                    "Each dataset item should include field 's2_text'"
                    " - str with s2 text."
                )

    @staticmethod
    def _shuffle_and_limit_index(index, limit):
        random.seed(42)
        random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
