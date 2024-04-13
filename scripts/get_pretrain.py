import os
import shutil
from pathlib import Path

import gdown

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

VIDEO_MODELS = {
    "frcnn_128_512.backbone.pth.tar": "13-T3nBnf21-lMKrV_XbH6Lf4vK2xU7lS",
    "lrw_snv05x_tcn1x.pth.tar": "197QXMxZ_fmsDxyvsqbDcV6XD7GUlaKHi",
}

CTCNET_MODELS = {"lrs2_best_model.pt": "1WtcpYYr8nMiIpJ1epnuGNk2DtiacUXDf"}


def get_pretrain():
    """
    Download pretrain video/ss models
    """
    data_path = ROOT_PATH / "data" / "pretrain"
    data_path.mkdir(exist_ok=True, parents=True)

    video_path = data_path / "video"
    ctcnet_path = data_path / "ctcnet"

    video_path.mkdir(exist_ok=True, parents=True)
    ctcnet_path.mkdir(exist_ok=True, parents=True)

    for key, id in VIDEO_MODELS.items():
        gdown.download(id=id)
        shutil.move(str(ROOT_PATH / key), str(video_path / key))

    for key, id in CTCNET_MODELS.items():
        gdown.download(id=id)
        shutil.move(str(ROOT_PATH / key), str(ctcnet_path / key))


if __name__ == "__main__":
    get_pretrain()
