import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
INFERENCE_PATH = ROOT_PATH / "inference.py"
SCRIPT_PATH = ROOT_PATH / "scripts" / "calculate_metrics.py"
ROOT_CONFIG_PATH = ROOT_PATH / "src" / "configs" / "model" / "inference.yaml"


def inference_and_calculate_all(dataset_name, checkpoints_path):
    print(checkpoints_path)

    inference_cmd = f"python3 {INFERENCE_PATH} model=inference datasets={dataset_name}"
    inference_cmd = inference_cmd + f" inferencer.save_path={dataset_name}"
    metrics_cmd = f"python3 {SCRIPT_PATH} -d={dataset_name}"

    for checkpoint_dir in os.listdir(checkpoints_path):
        dir_path = Path(checkpoints_path) / checkpoint_dir
        model_path = str(dir_path / "model_best.pth")
        config_path = str(dir_path / "config.yaml")

        # save config to load in from script
        config = OmegaConf.load(config_path)
        model_config = config.model
        OmegaConf.save(model_config, str(ROOT_CONFIG_PATH))

        model_inference_cmd = (
            inference_cmd + f" +inferencer.from_pretrained={model_path}"
        )
        model_metrics_cmd = metrics_cmd + f" -s={checkpoint_dir}"

        # run inference
        os.system(model_inference_cmd)

        # calculate metrics
        os.system(model_metrics_cmd)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Inference all models from dir and calculate all metrics on a given saved dataset"
    )
    args.add_argument(
        "-d",
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name inside data dir (default: None)",
    )

    args.add_argument(
        "-c",
        "--checkpoints_path",
        default=None,
        type=str,
        help="Path to all model checkpoints (default: None)",
    )

    args = args.parse_args()

    inference_and_calculate_all(args.dataset_name, args.checkpoints_path)
