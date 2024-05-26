import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from thop import profile

from src.trainer import Inferencer
from src.utils.data_utils import get_dataloaders
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    OmegaConf.set_struct(config, False)
    if config.model.get("asr_model") is not None:
        # change n_tokens
        config.model.asr_model.n_tokens = len(text_encoder)
    OmegaConf.set_struct(config, True)

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    print(model)

    model.eval()

    batch = next(iter(dataloaders["test"]))
    outputs = model(**batch)
    batch.update(outputs)
    # print(batch)

    mouth_emb = model.video_model(batch["s_video"])

    full_inputs = (batch["mix_audio"], batch["s_video"], batch["s_audio_length"])
    video_inputs = (batch["s_video"],)
    asr_inputs = (batch["fused_feats"], batch["s_audio_length"])
    ss_inputs = (batch["mix_audio"], mouth_emb)

    full_macs, _ = profile(model, full_inputs, verbose=False)

    video_macs, _ = profile(model.video_model, video_inputs, verbose=False)

    asr_macs, _ = profile(model.asr_model, asr_inputs, verbose=False)

    ss_macs, _ = profile(model.ss_model, ss_inputs, verbose=False)

    print(f"MACs (G): {full_macs / 10**9}")
    print(f"Video MACs (G): {video_macs / 10**9}")
    print(f"ASR MACs (G): {asr_macs / 10**9}")
    print(f"SS MACs (G): {ss_macs / 10**9}")


if __name__ == "__main__":
    main()
