import argparse

import torch


def get_state_dict(checkpoint_path, output_path):
    data = torch.load(checkpoint_path)

    state_dict = data["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("ss_model"):
            continue
        new_k = k[len("ss_model.") :]
        new_state_dict[new_k] = v

    # for consistency
    result = {"state_dict": new_state_dict}

    torch.save(result, output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Get SS state dict from full AVSSR checkpoint"
    )
    args.add_argument(
        "-c",
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to checkpoint (default: None)",
    )
    args.add_argument(
        "-o",
        "--output_path",
        default=None,
        type=str,
        help="Output path (default: None)",
    )

    args = args.parse_args()

    assert args.checkpoint_path is not None, "Provide path to checkpoint"
    assert args.output_path is not None, "Provide output path"

    get_state_dict(args.checkpoint_path, args.output_path)
