import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

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

    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        text_encoder=text_encoder,
    )

    inferencer.run_inference()


if __name__ == "__main__":
    main()
