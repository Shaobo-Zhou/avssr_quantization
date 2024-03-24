from typing import List
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import torch


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = defaultdict(list)

    # n_src == 1 is only supported for now
    for elem in dataset_items:
        result_batch["s_audio"].append(elem["s_audio"])
        result_batch["mix_audio"].append(elem["mix_audio"])
        result_batch["s_video"].append(elem["s_video"])
        result_batch["s_tokens"].append(elem["s_tokens"])
        result_batch["s_text"].append(elem["s_text"])
        result_batch["s_tokens_length"].append(elem["s_tokens_length"])
        result_batch["s_audio_length"].append(elem["s_audio_length"])

    result_batch["s_audio"] = torch.cat(result_batch["s_audio"], dim=0)
    result_batch["mix_audio"] = torch.cat(result_batch["mix_audio"], dim=0)
    result_batch["s_video"] = torch.cat(result_batch["s_video"], dim=0)
    result_batch["s_tokens"] = pad_sequence(result_batch["s_tokens"], batch_first=True)
    result_batch["s_tokens_length"] = torch.tensor(result_batch["s_tokens_length"])
    result_batch["s_audio_length"] = torch.tensor(result_batch["s_audio_length"])

    return result_batch
