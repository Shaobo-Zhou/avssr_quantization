import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer import Trainer
from src.utils.data_utils import get_dataloaders
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="src/configs", config_name="example")
def main(config):
    set_random_seed(config.trainer.seed)

    logger = setup_saving_and_logging(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

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
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "val": [], "test": []}
    for metric_type in ["train", "val", "test"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # build optimizer, learning rate scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if model.__class__.__name__ == "ExampleModel":
        params_dict = filter(lambda p: p.requires_grad, model.parameters())
    else:
        optimizer_params = OmegaConf.to_container(config.optimizer_params, resolve=True)
        ss_params = model.ss_model.parameters()
        asr_params = model.asr_model.parameters()
        params_dict = [
            {"params": asr_params, **optimizer_params["asr"]},
        ]
        if model.train_ss_model:
            params_dict.append({"params": ss_params, **optimizer_params["ss"]})
        if model.ss_teacher_proj is not None:
            params_dict.append(
                {"params": model.ss_teacher_proj.parameters(), **optimizer_params["ss"]}
            )
        if model.train_video_model:
            video_params = model.video_model.parameters()
            params_dict.append(
                {"params": video_params, **optimizer_params.get("video", {})}
            )
    optimizer = instantiate(config.optimizer, params=params_dict, _convert_="object")
    logger.info(optimizer)
    lr_scheduler = instantiate(
        config.lr_scheduler, optimizer=optimizer, _convert_="object"
    )
    logger.info(lr_scheduler)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        text_encoder=text_encoder,
    )

    trainer.train()


if __name__ == "__main__":
    main()
